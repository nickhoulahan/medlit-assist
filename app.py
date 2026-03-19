import tempfile
import re
import wave

import chainlit as cl
from chainlit.input_widget import Switch
from langchain_core.messages import AIMessage, HumanMessage

from src.asr.asr_model import ASRModel
from src.medlit_agent.agent.agent import OllamaAgent
from src.medlit_agent.tools.tools import tools
from src.tts.tts_model import TTSModel


def _clean_text_for_tts(text: str) -> str:
    """clean markdown-heavy responses to natural sounding words for TTS."""
    cleaned = text
    cleaned = re.sub(r"```[\s\S]*?```", " ", cleaned)
    cleaned = re.sub(r"https?://\S+", " ", cleaned) # remove URLs
    cleaned = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", cleaned) # markdown links
    cleaned = re.sub(r"(?m)^\s{0,3}(?:#{1,6}\s*|>\s*|[-*+]\s+|\d+\.\s+)", "", cleaned) # markdown syntax
    cleaned = re.sub(r":[a-zA-Z0-9_+\-]+:", " ", cleaned) # emojis like :smile:
    cleaned = cleaned.translate(str.maketrans("", "", "*_~`")) # markdown chars
    cleaned = cleaned.translate(
        str.maketrans("", "", "📄😊💡🔎🔍📚📖🧪🩺✅❌") # particular emojis
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip() # collapse whitespace
    return cleaned


def _extract_tool_status_info(streamed_text: str) -> str:
    """show user tool status lines visible after swapping in validated content."""
    status_lines = []
    for line in streamed_text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("🔎", "📄", "📚")):
            status_lines.append(stripped)

    return "\n\n".join(status_lines)


@cl.on_chat_start
async def start():
    agent = OllamaAgent(
        model="qwen3:8b",
        tools=tools,
        temperature=0.0,
        stream_chunk_size=18,
        stream_chunk_delay=0.025,
    )
    asr_model = ASRModel(model_name="openai/whisper-large-v3")
    tts_model = TTSModel()

    cl.user_session.set("agent", agent)
    cl.user_session.set("asr_model", asr_model)
    cl.user_session.set("tts_model", tts_model)
    cl.user_session.set("chat_history", [])
    settings = await cl.ChatSettings(
        [Switch(id="TTS_enabled", label="Enable Spoken Response", initial=False)]
    ).send()

    cl.user_session.set("TTS_enabled", settings["TTS_enabled"])


@cl.on_settings_update
async def on_settings_update(settings: dict):
    tts_enabled = bool(settings.get("TTS_enabled", False))
    cl.user_session.set("TTS_enabled", tts_enabled)


async def _handle_user_text_input(user_text: str) -> None:
    agent = cl.user_session.get("agent")

    chat_history = cl.user_session.get("chat_history", [])

    msg = cl.Message(content="")
    await msg.send()

    full_response = ""
    async for chunk in agent.astream(user_text, chat_history):
        if chunk:
            full_response += chunk
            await msg.stream_token(chunk)

    # after token streaming, replace response with schema-validated markdown when available
    # this allows a streaming response with after-the-fact schema validation
    validated_response = getattr(agent, "last_validated_response", None)
    if validated_response:
        tool_status_info = _extract_tool_status_info(full_response)
        follow_up_prompt = ""
        if "💡 *Any other follow-up questions? Just ask!*" in full_response:
            follow_up_prompt = "\n\n---\n\n💡 *Any other follow-up questions? Just ask!*"

        combined_parts = []
        if tool_status_info:
            combined_parts.append(tool_status_info)
        combined_parts.append(validated_response)

        full_response = "\n\n".join(combined_parts) + follow_up_prompt
        msg.content = full_response

    await msg.update()

    await _send_tts_audio_if_enabled(full_response, msg)

    chat_history.append(HumanMessage(content=user_text))
    chat_history.append(AIMessage(content=full_response))
    cl.user_session.set("chat_history", chat_history)


async def _send_tts_audio_if_enabled(response_text: str, response_message: cl.Message) -> None:
    if not response_text or not response_text.strip():
        return

    tts_enabled = cl.user_session.get("TTS_enabled", False)
    if not tts_enabled:
        return

    tts_model = cl.user_session.get("tts_model")
    if tts_model is None:
        await cl.Message(
            content="TTS model is not initialized. Please refresh the chat."
        ).send()
        return
    
    # inform user that spoken response is being generated
    while_waiting_message = cl.Message(content="Generating spoken response...")
    await while_waiting_message.send()

    try:
        tts_text = _clean_text_for_tts(response_text)
        if not tts_text:
            return

        wav_bytes, _ = tts_model.synthesize_speech_wav_bytes(tts_text)
        audio = cl.Audio(
            content=wav_bytes,
            name="medlit-response.wav",
            mime="audio/wav",
            auto_play=True,
        )
        # attach audio to the original assistant response to keep copy behavior on full text
        existing_elements = list(response_message.elements or [])
        response_message.elements = [*existing_elements, audio]
        await response_message.update()
    except Exception as e:
        await cl.Message(content=f"Spoken response error: please try again later.").send()
    finally:
        try:
            # remove the waiting message
            await while_waiting_message.remove()
        except Exception:
            pass


@cl.on_audio_start
async def on_audio_start():
    cl.user_session.set("audio_buffer", [])
    cl.user_session.set("audio_sr", 24000)  # Chainlit default commonly 24k
    return True


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    # get raw PCM16 bytes from each chunk.
    buffer = cl.user_session.get("audio_buffer")
    if buffer is not None:
        buffer.append(chunk.data)

    sample_rate = getattr(chunk, "sample_rate", None)
    if sample_rate:
        cl.user_session.set("audio_sr", sample_rate)


@cl.on_audio_end
async def on_audio_end():
    buffer = cl.user_session.get("audio_buffer")
    if not buffer:
        await cl.Message(content="No audio detected. Check mic permissions.").send()
        return

    pcm_bytes = b"".join(buffer)
    sample_rate = cl.user_session.get("audio_sr", 24000)

    # strategy is to write a temp wav file with PCM16 data for ASR transcription
    # once used, wav file is deleted

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        wav_path = tmp_file.name

    try:
        with wave.open(wav_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # PCM16
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_bytes)

        try:
            asr_model = cl.user_session.get("asr_model")
            # catch if asr model is not initialized
            if asr_model is None:
                await cl.Message(
                    content="ASR model is not initialized. Please refresh the chat."
                ).send()
                return

            transcript = asr_model.generate_text_response(
                wav_path,
                generate_kwargs={"language": "en", "task": "transcribe"},
            )
        except Exception as exc:
            await cl.Message(content=f"ASR error: {exc}").send()
            return
    finally:
        try:
            import os

            os.remove(wav_path)
        except Exception:
            pass

    if not transcript:
        await cl.Message(content="I did not catch that. Please try again.").send()
        return

    # echo transcript back as a user messsage, then use as input to agent
    await cl.Message(content=transcript, author="You", type="user_message").send()
    await _handle_user_text_input(transcript)


@cl.on_message
async def main(message: cl.Message):
    await _handle_user_text_input(message.content)
