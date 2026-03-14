import chainlit as cl
import tempfile
import wave
from langchain_core.messages import AIMessage, HumanMessage

from src.asr.asr_model import ASRModel
from src.medlit_agent.agent.agent import OllamaAgent
from src.medlit_agent.tools.tools import tools


@cl.on_chat_start
async def start():
    agent = OllamaAgent(model="qwen3:8b", tools=tools, temperature=0.0)
    asr_model = ASRModel(model_name="openai/whisper-large-v3")

    cl.user_session.set("agent", agent)
    cl.user_session.set("asr_model", asr_model)
    cl.user_session.set("chat_history", [])


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

    await msg.update()

    chat_history.append(HumanMessage(content=user_text))
    chat_history.append(AIMessage(content=full_response))
    cl.user_session.set("chat_history", chat_history)


@cl.on_audio_start
async def on_audio_start():
    cl.user_session.set("audio_buffer", [])
    cl.user_session.set("audio_sr", 24000)  # Chainlit default commonly 24k
    return True


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    # Collect raw PCM16 bytes from each chunk.
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
                await cl.Message(content="ASR model is not initialized. Please refresh the chat.").send()
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
