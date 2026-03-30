import argparse
import asyncio
import os
import os
from datetime import datetime

import aiohttp
import pytz
from dotenv import load_dotenv
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer, VADParams
from pipecat.frames.frames import LLMMessagesFrame, TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.aggregators.anthropic_llm_context import AnthropicLLMContext
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.transports.network.websocket_client import (
    WebsocketClientParams,
    WebsocketClientTransport,
)

from config.persona_utils import PersonaManager
from config.prompts import DEFAULT_SYSTEM_PROMPT
from meetingbaas_pipecat.utils.logger import configure_logger
import sys
import logging


from pipecat.services.llm_service import FunctionCallParams

load_dotenv(override=True)

logger = configure_logger()

# Ensure logs are flushed immediately and are human-readable
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s | %(name)s:%(funcName)s:%(lineno)d | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)
logger.handlers = [handler]
logger.propagate = False

# Function to log and flush
def log_and_flush(level, msg):
    logger.log(level, msg)
    for h in logger.handlers:
        h.flush()

# Function tool implementations
async def get_weather(params: FunctionCallParams):
    """Get the current weather for a location."""
    arguments = params.arguments
    location = arguments["location"]
    format = arguments["format"]
    unit = (
        "m" if format == "celsius" else "u"
    )

    url = f"https://wttr.in/{location}?format=%t+%C&{unit}"

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                weather_data = await response.text()
                await params.result_callback(
                    f"The weather in {location} is currently {weather_data} ({format.capitalize()})."
                )
            else:
                await params.result_callback(
                    f"Failed to fetch the weather data for {location}."
                )


async def get_time(params: FunctionCallParams):
    """Get the current time for a location."""
    arguments = params.arguments
    location = arguments["location"]

    try:
        timezone = pytz.timezone(location)
        current_time = datetime.now(timezone)
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        await params.result_callback(f"The current time in {location} is {formatted_time}.")
    except pytz.UnknownTimeZoneError:
        await params.result_callback(
            f"Invalid location specified. Could not determine time for {location}."
        )

async def main(
    meeting_url: str = "",
    persona_name: str = "Meeting Bot",
    entry_message: str = "Hello, I am the meeting bot",
    bot_image: str = "",
    streaming_audio_frequency: str = "24khz",
    websocket_url: str = "",
    enable_tools: bool = True,
):
    from pipecat.utils.asyncio import TaskManager
    TaskManager.set_event_loop(TaskManager, asyncio.get_running_loop())
    
    log_and_flush(logging.INFO, f"[STARTUP] MeetingBaas bot launching with persona: {persona_name}")
    load_dotenv()

    if not websocket_url:
        log_and_flush(logging.ERROR, "[ERROR] WebSocket URL not provided")
        return
    log_and_flush(logging.INFO, f"[CONFIG] Using WebSocket URL: {websocket_url}")
    parts = websocket_url.split("/")
    expected_local_port = os.getenv("PORT", "7014")
    if "localhost" in websocket_url and f":{expected_local_port}/pipecat/" in websocket_url:
        bot_id = parts[-1] if len(parts) > 3 else "unknown"
    elif "ngrok.io" in websocket_url:
        bot_id = parts[-1] if len(parts) > 3 and parts[-2] == "pipecat" else "unknown"
    else:
        bot_id = parts[-1] if len(parts) > 3 else "unknown"
    logger.info(f"Using bot ID: {bot_id}")

    output_sample_rate = 24000 if streaming_audio_frequency == "24khz" else 16000
    vad_sample_rate = 16000
    log_and_flush(logging.INFO, f"[CONFIG] Audio frequency: {streaming_audio_frequency} (output: {output_sample_rate}, VAD: {vad_sample_rate})")

    print("Event loop set for Pipecat:", asyncio.get_running_loop())

    transport = WebsocketClientTransport(
        uri=websocket_url,
        params=WebsocketClientParams(
            audio_out_sample_rate=output_sample_rate,
            audio_out_enabled=True,
            add_wav_header=False,
            audio_in_enabled=True,
            vad_analyzer=SileroVADAnalyzer(
                sample_rate=16000,
                params=VADParams(
                    threshold=0.5,
                    min_speech_duration_ms=250,
                    min_silence_duration_ms=100,
                    min_volume=0.6,
                ),
            ),
            audio_in_passthrough=True,
            serializer=ProtobufFrameSerializer(),
            timeout=300,
        ),
    )
    log_and_flush(logging.INFO, "[TRANSPORT] WebSocket transport initialized")

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        log_and_flush(logging.INFO, "[WEBSOCKET] Client connected to WebSocket server")
        
    @transport.event_handler("on_client_disconnected") 
    async def on_client_disconnected(transport, client):
        log_and_flush(logging.INFO, "[WEBSOCKET] Client disconnected from WebSocket server")

    @transport.event_handler("on_connection_established")
    async def on_connection_established(transport):
        log_and_flush(logging.INFO, "[WEBSOCKET] WebSocket connection established successfully")
        
    @transport.event_handler("on_connection_error")
    async def on_connection_error(transport, error):
        log_and_flush(logging.ERROR, f"[WEBSOCKET] Connection error: {error}")

    persona_manager = PersonaManager()
    persona = persona_manager.get_persona(persona_name)
    if not persona:
        log_and_flush(logging.ERROR, f"[ERROR] Persona '{persona_name}' not found")
        return
    log_and_flush(logging.INFO, f"[PERSONA] Loaded persona: {persona_name}")

    additional_content = persona.get("additional_content", "")

    # Use Deepgram Aura TTS
    voice_id = persona.get("deepgram_voice_id") or os.getenv("DEEPGRAM_VOICE_ID", "aura-asteria-en")
    log_and_flush(logging.INFO, f"[PERSONA] Using Deepgram voice: {voice_id}")

    tts = DeepgramTTSService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        voice=voice_id,
        sample_rate=output_sample_rate,
    )
    log_and_flush(logging.INFO, f"[TTS] Deepgram TTS initialized with sample_rate={output_sample_rate}, voice={voice_id}")

    llm = AnthropicLLMService(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        model="claude-sonnet-4-20250514",
    )
    log_and_flush(logging.INFO, f"[LLM] Anthropic Claude initialized with model=claude-sonnet-4-20250514")

    if enable_tools:
        log_and_flush(logging.INFO, "[TOOLS] Registering function tools")
        llm.register_function("get_weather", get_weather)
        llm.register_function("get_time", get_time)

        weather_function = FunctionSchema(
            name="get_weather",
            description="Get the current weather",
            properties={
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "format": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use. Infer this from the users location.",
                },
            },
            required=["location", "format"],
        )

        time_function = FunctionSchema(
            name="get_time",
            description="Get the current time for a specific location",
            properties={
                "location": {
                    "type": "string",
                    "description": "The location for which to retrieve the current time",
                },
            },
            required=["location"],
        )

        tools = ToolsSchema(standard_tools=[weather_function, time_function])
    else:
        log_and_flush(logging.INFO, "[TOOLS] Function tools are disabled")
        tools = None

    language = persona.get("language_code", "en-US")

    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        encoding="linear16" if streaming_audio_frequency == "16khz" else "linear24",
        sample_rate=output_sample_rate,
        language=language,
    )

    bot_name = persona_name or "Bot"
    system_content = persona["prompt"]

    if additional_content:
        system_content += f"\n\nYou are {persona_name}\n\n{DEFAULT_SYSTEM_PROMPT}\n\n"
        system_content += "You have the following additional context. USE IT TO INFORM YOUR RESPONSES:\n\n"
        system_content += additional_content

    messages = [{"role": "system", "content": system_content}]

    if enable_tools and tools:
        context = AnthropicLLMContext(messages, tools)
    else:
        context = AnthropicLLMContext(messages)

    aggregator_pair = llm.create_context_aggregator(context)
    user_aggregator = aggregator_pair.user()
    assistant_aggregator = aggregator_pair.assistant()

    pipeline = Pipeline([
        transport.input(),
        stt,
        user_aggregator,
        llm,
        tts,
        assistant_aggregator,
        transport.output(),
    ])

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True, check_dangling_tasks=True))
    runner = PipelineRunner()

    if entry_message:
        initial_message = {"role": "user", "content": entry_message}
        async def queue_initial_message():
            await asyncio.sleep(2)
            await task.queue_frames([LLMMessagesFrame([initial_message])])
        asyncio.create_task(queue_initial_message())

    try:
        log_and_flush(logging.INFO, "[RUN] Starting pipeline runner...")
        await runner.run(task)
    except Exception as e:
        log_and_flush(logging.ERROR, f"[ERROR] Exception in pipeline: {e}")
        import traceback
        log_and_flush(logging.ERROR, f"[ERROR] Traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a MeetingBaas bot")
    parser.add_argument("--meeting-url", help="URL of the meeting to join")
    parser.add_argument("--persona-name", default="Meeting Bot", help="Name to display for the bot")
    parser.add_argument("--entry-message", default="Hello, I am the meeting bot", help="Message to send when joining")
    parser.add_argument("--bot-image", default="", help="URL for bot avatar")
    parser.add_argument("--streaming-audio-frequency", default="16khz", choices=["16khz", "24khz"], help="Audio frequency")
    parser.add_argument("--websocket-url", help="Full WebSocket URL to connect to")
    parser.add_argument("--enable-tools", action="store_true", help="Enable function tools")
    parser.add_argument("--client-id", help="Internal client ID for the bot")
    parser.add_argument("--persona-data-json", help="Persona data as JSON string")
    parser.add_argument("--api-key", help="API key for authentication")
    parser.add_argument("--meetingbaas-bot-id", help="MeetingBaas bot ID")

    args = parser.parse_args()

    persona_name = args.persona_name
    if args.persona_data_json:
        try:
            import json
            persona_data = json.loads(args.persona_data_json)
            from config.persona_utils import PersonaManager
            pm = PersonaManager()
            for folder_name, data in pm.personas.items():
                if data.get("name") == persona_data.get("name"):
                    persona_name = folder_name
                    break
        except Exception as e:
            print(f"Error parsing persona data JSON: {e}")

    asyncio.run(
        main(
            meeting_url=args.meeting_url,
            persona_name=persona_name,
            entry_message=args.entry_message,
            bot_image=args.bot_image,
            streaming_audio_frequency=args.streaming_audio_frequency,
            websocket_url=args.websocket_url,
            enable_tools=args.enable_tools,
        )
    )
