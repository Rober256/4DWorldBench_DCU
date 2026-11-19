import os
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoModel, AutoTokenizer
from transformers import AutoProcessor, AutoModel, AutoTokenizer
import keye_vl_utils
import qwen_vl_utils
import torch
import json
import datetime
import tqdm
from runtime_config import keye_model_path
from metric.keye_runtime import keye_max_new_tokens, keye_video_caption_prompt, keye_video_content


_caption_model = None
_caption_processor = None
_caption_model_device = None


def preload_keye_caption_model(device=None):
    """Load the caption model once, at an explicit and measurable call site."""
    global _caption_model, _caption_processor, _caption_model_device
    requested_device = device or os.environ.get("CAPTION_DEVICE", "cuda:0")
    if _caption_model is None or _caption_processor is None:
        _caption_model = AutoModel.from_pretrained(
            str(keye_model_path()),
            torch_dtype="auto",
            device_map=requested_device,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
        min_pixels = 512 * 28 * 28
        max_pixels = 7680 * 28 * 28
        _caption_processor = AutoProcessor.from_pretrained(
            str(keye_model_path()),
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            trust_remote_code=True,
        )
        _caption_model.eval()
        _caption_model_device = requested_device
    return _caption_model, _caption_processor


def _extract_keye_final_text(decoded_text):
    """Normalize Keye direct, thinking-tag, answer-tag, and boxed outputs."""
    text = decoded_text[0] if isinstance(decoded_text, list) else str(decoded_text)
    if "<answer>" in text:
        answer = text.split("<answer>", 1)[1]
        return answer.split("</answer>", 1)[0].strip()
    if "</analysis>" in text:
        return text.split("</analysis>", 1)[1].strip()
    if "boxed{" in text:
        boxed = text.rsplit("boxed{", 1)[1].strip()
        return boxed[:-1].strip() if boxed.endswith("}") else boxed
    return text.strip()


def Qwen_describe_image(image_path, device="cuda"):
    model, processor = preload_keye_caption_model()
    # Messages containing a local video path and a text query
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                    # "max_pixels": 1920 * 1080,
                },
                {"type": "text", "text": "You are an useful assistant responsible for observing the image and describing the main physical phenomena and events in it. Keep output concise and focused. Only describe visible events, don't describe any atmosphere or ambiance, don't make assumptions."}
                # {"type": "text", "text": "Describe the physical phenomena that are visible in the video concisely without making too many assumptions."}
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    image_inputs, video_inputs = qwen_vl_utils.process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        # **video_kwargs,
    )
    inputs = inputs.to(device)

    # Inference
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

def Keye_describe_image(image_path, device="cuda"):
    model, processor = preload_keye_caption_model()
    # Messages containing a local video path and a text query
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": "In a single paragraph, provide a purely objective description of the image. Detail only the tangible entities visible—such as people and objects—along with their key attributes (e.g., color, shape, position) and any concrete actions taking place. The description must be concise and strictly based on visual evidence. Do not describe, interpret, or infer any emotions, atmosphere, or abstract concepts. Make no assumptions or associations about anything beyond what is physically present in the image."}
                # {"type": "text", "text": "Describe the physical phenomena that are visible in the video concisely without making too many assumptions."}
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, mm_processor_kwargs = keye_vl_utils.process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **mm_processor_kwargs,
    )
    inputs = inputs.to(device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=768)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return _extract_keye_final_text(output_text)

def Qwen_describe_video(video_path, fps, device="cuda"):
    model, processor = preload_keye_caption_model()
    # Messages containing a local video path and a text query
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 1920 * 1080,
                    "fps": fps,
                },
                {"type": "text", "text": "You are an useful assistant responsible for observing the video and summarizing the main phenomena and events in it. Focus on key events and interactions in the video and present them in chronological order. Don't describe any atmosphere or ambiance."}
                # {"type": "text", "text": "Describe the physical phenomena that are visible in the video concisely without making too many assumptions."}
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    image_inputs, video_inputs = qwen_vl_utils.process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        fps=fps,
        padding=True,
        return_tensors="pt",
        # **video_kwargs,
    )
    inputs = inputs.to(device)

    # Inference
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text

def Keye_describe_video_fps(video_path, fps, device="cuda"):
    model, processor = preload_keye_caption_model()
    # Messages containing a local video path and a text query
    messages = [
        {
            "role": "user",
            "content": [
                keye_video_content(video_path, fps=fps),
                {"type": "text", "text": keye_video_caption_prompt()}
                # {"type": "text", "text": "You are an useful assistant responsible for observing the video and summarizing the main phenomena and events in it. Focus on key events and interactions in the video and present them in chronological order. Don't describe any atmosphere or ambiance."}
                # {"type": "text", "text": "Describe the physical phenomena that are visible in the video concisely without making too many assumptions."}
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, mm_processor_kwargs = keye_vl_utils.process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **mm_processor_kwargs,
        fps=fps,
    )
    inputs = inputs.to(device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=keye_max_new_tokens())
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    if "<analysis>" in output_text[0]:
        output_text = output_text[0].split("</analysis>")[1]
    elif "<think>" in output_text[0]:
        start_index = output_text[0].find("<answer>")
        end_index = output_text[0].find("</answer>")
        output_text = output_text[0][start_index + len("<answer>"):end_index]
    return output_text

def Keye_describe_video(video_path, device="cuda", reference_prompt="", evaluation_questions=None):
    model, processor = preload_keye_caption_model()
    # Messages containing a local video path and a text query
    messages = [
        {
            "role": "user",
            "content": [
                keye_video_content(video_path),
                {
                    "type": "text",
                    "text": keye_video_caption_prompt(reference_prompt, evaluation_questions),
                }
                # {"type": "text", "text": "You are an useful assistant responsible for observing the video and summarizing the main phenomena and events in it. Focus on key events and interactions in the video and present them in chronological order. Don't describe any atmosphere or ambiance."}
                # {"type": "text", "text": "Describe the physical phenomena that are visible in the video concisely without making too many assumptions."}
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, mm_processor_kwargs = keye_vl_utils.process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **mm_processor_kwargs,
        # fps=fps,
    )
    inputs = inputs.to(device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=keye_max_new_tokens())
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return _extract_keye_final_text(output_text)

def Keye_answer_video(video_path, device="cuda"):
    model, processor = preload_keye_caption_model()
    # Messages containing a local video path and a text query
    messages = [
        {
            "role": "user",
            "content": [
                keye_video_content(video_path),
                {"type": "text", "text": keye_video_caption_prompt()}
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, mm_processor_kwargs = keye_vl_utils.process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **mm_processor_kwargs,
        # fps=fps,
    )
    inputs = inputs.to(device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=keye_max_new_tokens())
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    if "<analysis>" in output_text[0]:
        output_text = output_text[0].split("</analysis>")[1]
    elif "<think>" in output_text[0]:
        start_index = output_text[0].find("<answer>")
        end_index = output_text[0].find("</answer>")
        output_text = output_text[0][start_index + len("<answer>"):end_index]
    return output_text

def get_image_paths_from_directory(directory):
    supported_formats = ['jpg','png']  # Supported image formats
    img_paths = []
    
    for root, _, files in os.walk(directory):  # Traverse all subdirectories
        for file in files:
            if any(file.lower().endswith(fmt) for fmt in supported_formats):
                img_paths.append(os.path.join(root, file))  # Add full path to list
    
    return img_paths

def get_image_paths_from_directory_check(directory, exclude_file="output.jpg"):
    supported_formats = ['jpg','png']  # Supported image formats
    img_paths = []
    
    for root, _, files in os.walk(directory):  # Traverse all subdirectories
        for file in files:
            if any(file.lower().endswith(fmt) for fmt in supported_formats):
                full_path = os.path.join(root, file)
                # Check if in exclusion list
                if exclude_file and file == exclude_file:
                    img_paths.append(full_path)  # Add full path to list
    return img_paths

def get_video_paths_from_directory(directory):
    supported_formats = ['.mp4', '.avi', '.mov']  # Supported video formats
    video_paths = []
    
    for root, _, files in os.walk(directory):  # Traverse all subdirectories
        for file in files:
            if any(file.lower().endswith(fmt) for fmt in supported_formats):
                video_paths.append(os.path.join(root, file))  # Add full path to list
    
    return video_paths

def get_video_paths_from_directory_check(directory, exclude_file="output.mp4"):
    supported_formats = ['.mp4', '.avi', '.mov']  # Supported video formats
    video_paths = []
    
    for root, _, files in os.walk(directory):  # Traverse all subdirectories
        for file in files:
            if any(file.lower().endswith(fmt) for fmt in supported_formats):
                full_path = os.path.join(root, file)
                # Check if in exclusion list
                if exclude_file and file == exclude_file:
                    video_paths.append(full_path)  # Add full path to list
    return video_paths

def save_to_json(data, directory, filename=None):
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H:%M:%S")
        filename = f"output/condition_caption/{directory.split('/')[-2]}_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
