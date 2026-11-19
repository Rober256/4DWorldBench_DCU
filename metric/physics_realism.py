#!/usr/bin/env python3
"""
Physics reasoning evaluation script
Adapted to framework based on original gpt4o_quesgen_0827_for_vbenchsubsample.py logic
Keeping the original question generation and answering logic unchanged
"""

import json
import os
import sys
import warnings
from tqdm import tqdm
from time import sleep
from typing import List, Dict, Any, Tuple
from openai import OpenAI
from runtime_config import api_model_name
import json
from concurrent.futures import ThreadPoolExecutor, as_completed  # Added: concurrent execution

# Add path to import framework components
#sys.path.append('/data2/luowei/phywordbench')

try:
    from metric.utils import load_dimension_info
except ImportError:
    from .utils import load_dimension_info
from .timing import get_timer

try:
    from metric_compatibility import ensure_auxiliary_info_compatibility
    COMPATIBILITY_AVAILABLE = True
except ImportError:
    COMPATIBILITY_AVAILABLE = False

warnings.filterwarnings("ignore")

class PhysicsReasoningEvaluator:
    """Physics reasoning evaluator"""
    
    def __init__(self, device: str = "cpu", timer=None):
        """Initialize evaluator"""
        self.device = device
        self.timer = timer
        print("Using device: %s" % self.device)
        
        # Initialize OpenAI client (reads OPENAI_API_KEY and OPENAI_BASE_URL from env)
        self.client = OpenAI(
            base_url=os.environ.get('OPENAI_BASE_URL', ''),
            api_key=os.environ.get('OPENAI_API_KEY', ''),
        )
        self.model_type = api_model_name()

        # Physics dimensions definition (from original file)
        self.physics_dimensions = [
            "Fundamental Physics", 
            "Temporal & Spatial rationality", 
            "Physical Event Reasoning", 
            "Force & Motion", 
            "Optical Phenomena", 
            "Material Interaction & Transformation", 
            "Thermal & Phase Transition", 
            "Overall Description"
        ]
        
        # self.physics_dimensions = [
        #     "Fundamental Physics", 
        #     "Object Properties & Affordances", 
        #     "Spatial Reasoning", 
        #     "Force & Motion", 
        #     "Optical Phenomena", 
        #     "Material Interaction & Transformation", 
        #     "Thermal and Phase Transition", 
        #     "Overall Description"
        # ]
        
        # Physics reasoning evaluation prompt (based on original complex_prompt_v6)
        self.physics_prompt = (
            "You are a scientist who designs diagnostic yes/no questions about short real‑world scenarios. "
            "Given: (1) a natural‑language description of a real‑world scene, and (2) an ordered list of dimensions "
            "(e.g., Fundamental Physics, Optics, Material Interaction & Transformation, Force & Motion, Thermal & Phase Transition, etc.). "
            "Returns ten(10) questions in the order provided, with one to four questions per dimension. If there are too few dimensions to ask, more questions per dimension will be required."
            "Reasoning about reasonable phenomena that should occur in the real world based on the description and posing them as questions, such as a balloon should explode when a sharp object presses into it, water should boil at above 100 degrees, cheese should melt at high temperatures, etc."
            "You should reason about what should happen in the real world, each question should be crafted to be answerable solely by inspecting the description and foucus on visible phenomenon in the description without requiring external knowledge. "
            "If the input description does not contain some problem dimensions or can not design 'yes' answer question, skip generating questions for that part and move on to the next dimension and do not invent imaginary properties. "
            "Each question must stay strictly within its assigned dimension's scope. Avoid cross‑dimension leakage. "
            "Use present tense, neutral tone, and end each question with '(yes or no)'. "

            "Return a JSON array of objects, each with: "
            "{'dimension': '<dimension name>', "
            "'auxiliary_info': ['<one to four yes/no questions>']}. "
            "Preserve the dimension order. Validate: at most 8 objects, each auxiliary_info has one to four questions, "
            "all questions are dimension‑appropriate, observable, and answer 'yes'. "

            "Here are some in‑context examples:\n\n"
            "'questions': ["
            "{'dimension': 'Material Interaction & Transformation', "
            "'auxiliary_info': ["
            "'Does the blue and yellow paints visibly exist? (yes or no)', "
            "'Does the blue and yellow paints mix visibly during the stirring process? (yes or no)', "
            "'Does the blue and yellow disappear from the mixed paint? (yes or no)', "    
            "'Finally, does the paint become green? (yes or no)']},\n\n"
            "{'dimension': 'Force & Motion', "
            "'auxiliary_info': ["
            "'Does the toothpaste tube contact with hands? (yes or no)', "
            "'Does the toothpaste tube deform under stress? (yes or no)', "
            "'Does the toothpaste tube be compressed and the toothpaste be expelled out of the toothpaste tube? (yes or no)']},\n\n"
            "{'dimension': 'Thermal & Phase Transition', "
            "'auxiliary_info': ["
            "'Initially, is the river in a liquid state? (yes or no)', "
            "'Finally, does the river freeze? (yes or no)']}]\n\n"
        )    

        # self.prompt = (
        #     "You are a precise yes/no caption evidence checker.\n"
        #     "Goal: maximize TRUE positives while avoiding unsupported 'yes'.\n"
        #     "Input: one caption + one yes/no question about OBSERVABLE physical facts/events/states.\n"
        #     "Decision Rules (lenient mode):\n"
        #     "1. Use ONLY the caption text (but allow paraphrase / synonyms / reordering equivalence).\n"
        #     "2. A detail is SUPPORTED if its core entities & relations appear explicitly OR via clear synonym (e.g., 'cup' vs 'mug', 'holding' vs 'grasping').\n"
        #     "3. Ignore missing minor adjectives, intensifiers, colors, counts unless the question explicitly hinges on them.\n"
        #     "4. Multi‑clause question: answer 'yes' if EVERY core clause (main subject + main action/state + essential object) is supported; tolerate benign tense shifts.\n"
        #     "5. Answer 'no' ONLY if there is a required core element absent OR contradicted.\n"
        #     "6. If partially supported but missing a CORE relation (e.g., action absent), answer 'no'.\n"
        #     "7. Default bias: prefer 'yes' when evidence is reasonably present; avoid false 'no'.\n"
        #     "8. NEVER judge real‑world plausibility—only textual support.\n"
        #     "Output EXACT JSON ONLY: {\"answer\":\"yes\"} or {\"answer\":\"no\"}."
        # )

        self.prompt = "Please answer yes or no only for the following questions according to the caption."
        # self.prompt = "You are an expert at answering questions based on descriptions of generated videos, which may contain various physically unreasonable. Please answer yes or no only for the following question according to the caption. When answering, you should carefully check whether the main objects and behaviors of the question and caption are consistent."

        # Concurrency and rate-limiting controls
        self.max_workers = int(os.environ.get("PR_MAX_WORKERS", "10"))
        self.request_delay = float(os.environ.get("PR_REQUEST_DELAY", "0.5"))
        self.max_retries = int(os.environ.get("PR_MAX_RETRIES", "2"))
        self.max_consecutive_failures = int(os.environ.get("PR_MAX_CONSECUTIVE_FAILURES", "10"))

    def generate_physics_questions(self, description: str) -> List[Dict[str, Any]]:
        """
        Generate physics reasoning questions (keeping original logic)
        
        Args:
            description: Video description
            
        Returns:
            List[Dict]: List of generated questions
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_type,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an useful assistant that only outputs valid JSON format. Always use double quotes for keys and values, and never use single quotes or any extra text.The format should be: questions:[q1,q2,...]."
                    },
                    {"role": "user", "content": f"prompt:{self.physics_prompt}"},
                    {"role": "user", "content": f"dimension:{self.physics_dimensions}"},
                    {"role": "user", "content": f"description:{description}"},
                ],
                temperature=0,
            )
            
            input_string = response.choices[0].message.content
            input_string = input_string.strip().lstrip("```json").rstrip("```").strip()
            
            try:
                result = json.loads(input_string)
                if isinstance(result, dict) and 'questions' in result:
                    return result['questions']
                elif isinstance(result, list):
                    return result
                else:
                    print(f"Unexpected response format: {result}")
                    return []
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed: {e}")
                return []
                
        except Exception as e:
            print(f"Failed to generate physics questions: {e}")
            return []

    def answer_physics_question(self, question: str, caption: str) -> str:
        """
        Answer physics reasoning questions with retry and exponential backoff.
        Raises RuntimeError if all retries are exhausted.
        """
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_type,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an assistant that only outputs valid JSON format. Always use double quotes for keys and values, and never use single quotes or any extra text. Example: {\"answer\":\"yes\"} or {\"answer\":\"no\"}"
                        },
                        {"role": "user", "content": self.prompt},
                        {"role": "user", "content": f"Caption: {caption}"},
                        {"role": "user", "content": f"Question: {question}"},
                    ],
                    temperature=0,
                )

                input_string = response.choices[0].message.content
                input_string = input_string.strip().lstrip("```json").rstrip("```").strip()

                try:
                    parsed_data = json.loads(input_string)

                    if isinstance(parsed_data, dict) and 'answer' in parsed_data:
                        answer_text = parsed_data['answer'].lower()
                    elif isinstance(parsed_data, str):
                        answer_text = parsed_data.lower()
                    else:
                        answer_text = 'null'

                    if answer_text == 'yes':
                        return 'yes'
                    elif answer_text == 'no':
                        return 'no'
                    else:
                        return 'null'

                except json.JSONDecodeError as e:
                    # Parse errors are not server-side issues, don't retry
                    print(f"Failed to parse answer: {e}")
                    return 'null'

            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    delay = (2 ** attempt) * self.request_delay
                    print(f"API call failed (attempt {attempt+1}/{self.max_retries+1}): {e}, retrying in {delay:.1f}s...")
                    sleep(delay)
                else:
                    print(f"API call failed after {self.max_retries+1} attempts: {e}")

        raise RuntimeError(f"Failed to answer question after {self.max_retries+1} attempts: {last_error}")

    # Added: extract pure question list from generated results
    def _extract_questions_from_generated(self, physics_questions_raw: List[Dict[str, Any]]) -> List[str]:
        questions: List[str] = []
        for q_group in physics_questions_raw or []:
            if isinstance(q_group, dict) and 'auxiliary_info' in q_group:
                questions.extend(q_group['auxiliary_info'])
        # Remove empty/non-string and deduplicate (deduplicate by lowercase trimmed trailing whitespace)
        seen = set()
        cleaned = []
        for q in questions:
            if isinstance(q, str):
                k = q.strip().lower()
                if k and k not in seen:
                    seen.add(k)
                    cleaned.append(q.strip())
        return cleaned

    # Ensure the configured number of questions (10 by default; lower for smoke tests).
    def _ensure_ten_questions(self, prompt_text: str, base_questions: List[str]) -> List[str]:
        target = max(1, int(os.environ.get("PR_TARGET_QUESTIONS", "10")))
        # Initial cleaning and deduplication
        uniq = []
        seen = set()
        for q in base_questions or []:
            if isinstance(q, str):
                k = q.strip().lower()
                if k and k not in seen:
                    seen.add(k)
                    uniq.append(q.strip())
        # If insufficient, retry up to two rounds to generate more
        retries = 0
        while len(uniq) < target and retries < 2:
            retries += 1
            physics_questions_raw = self.generate_physics_questions(prompt_text)
            more = self._extract_questions_from_generated(physics_questions_raw)
            for q in more:
                k = q.strip().lower()
                if k and k not in seen:
                    seen.add(k)
                    uniq.append(q.strip())
            # If still insufficient, continue looping until reaching target or retry limit
        # Truncate to the configured target.
        if len(uniq) > target:
            uniq = uniq[:target]
        return uniq

    def evaluate_physics_reasoning(self, prompt_dict_ls: List[Dict[str, Any]]) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Evaluate physics reasoning ability
        
        Args:
            prompt_dict_ls: List of input data
            
        Returns:
            Tuple[float, List[Dict]]: (average score, detailed results)
        """
        total_score = 0.0
        valid_items = 0
        detailed_results = []
        
        for prompt_dict in tqdm(prompt_dict_ls):
            # Get video information
            print(f"===== Evaluating video list: {prompt_dict} =====")
            video_list = prompt_dict.get('video_list', [])
            prompt_text = prompt_dict.get('prompt', '')
            auxiliary_info = prompt_dict.get('auxiliary_info', [])
            
            for video_path in video_list:
                video_timing = self.timer.start_video(video_path) if self.timer else None
                print(f"\n===== Evaluating video: {video_path} =====")
                
                # Check if there are pre-generated questions
                if auxiliary_info:
                    # Use pre-generated questions and ensure 10 questions (generate more if insufficient, truncate if excess)
                    print(f"Using {len(auxiliary_info)} pre-generated physics reasoning questions")
                    questions_init = [q for q in auxiliary_info if isinstance(q, str)]
                    questions = self._ensure_ten_questions(prompt_text, questions_init)
                else:
                    # Dynamically generate questions (one generation + additional if insufficient), ensure 10 questions
                    print("Dynamically generating physics reasoning questions...")
                    physics_questions_raw = self.generate_physics_questions(prompt_text)
                    questions_first = self._extract_questions_from_generated(physics_questions_raw)
                    questions = self._ensure_ten_questions(prompt_text, questions_first)
                
                if not questions:
                    print("Failed to generate valid physics questions, skipping this video")
                    if video_timing is not None:
                        video_timing.status = "failed"
                        video_timing.error = "failed to generate valid physics questions"
                        self.timer.finish_video(video_timing)
                    continue

                print(f"Number of physics reasoning questions determined for evaluation: {len(questions)} (target: {max(1, int(os.environ.get('PR_TARGET_QUESTIONS', '10')))})")
                
                # Initialize results
                video_detail = {
                    "video_path": video_path,
                    "prompt": prompt_text,
                    "questions": questions,
                    "answers": [],
                    "question_details": [],
                }
                
                # Get video description
                print("Getting video description...")
                from caption_method.VideoImg_caption import Keye_describe_video
                use_reference = os.environ.get("KEYE_USE_REFERENCE_PROMPT", "0").lower() in {
                    "1", "true", "yes"
                }
                guide_with_questions = os.environ.get(
                    "KEYE_GUIDE_WITH_QUESTIONS", "0"
                ).lower() in {"1", "true", "yes"}
                self.caption = Keye_describe_video(
                    video_path,
                    device=self.device,
                    reference_prompt=prompt_text if use_reference else "",
                    evaluation_questions=questions if guide_with_questions else None,
                )
                print(f"Video description: {self.caption}")

                # Answer questions concurrently with rate limiting and circuit breaker
                total_questions = len(questions)
                answers = [None] * total_questions
                details = [None] * total_questions
                print(f"Starting to answer {total_questions} questions concurrently, max concurrency: {self.max_workers}, request delay: {self.request_delay}s")
                consecutive_failures = 0
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # Stagger submissions to avoid flooding the API
                    future_to_idx = {}
                    for idx, q in enumerate(questions):
                        if idx > 0:
                            sleep(self.request_delay)
                        future_to_idx[executor.submit(self.answer_physics_question, q, self.caption)] = idx

                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            answer = future.result()
                            consecutive_failures = 0
                        except Exception as e:
                            print(f"Concurrent answer exception idx={idx}: {e}")
                            answer = 'null'
                            consecutive_failures += 1
                            if consecutive_failures >= self.max_consecutive_failures:
                                print(f"ERROR: {consecutive_failures} consecutive API failures, aborting evaluation to avoid triggering risk controls")
                                for f in future_to_idx:
                                    f.cancel()
                                raise RuntimeError(
                                    f"Aborted: {consecutive_failures} consecutive API call failures. "
                                    "Consider increasing PR_REQUEST_DELAY, reducing PR_MAX_WORKERS, or checking the API service."
                                )
                        answers[idx] = answer
                        details[idx] = {
                            "question": questions[idx],
                            "video_caption": self.caption,
                            "answer": answer,
                            "is_correct": answer == 'yes'
                        }

                # Write back and score
                video_detail["answers"] = answers
                video_detail["question_details"] = details

                correct_answers = sum(1 for a in answers if a == 'yes')
                total_questions = len(questions)
                if total_questions > 0:
                    video_score = correct_answers / total_questions
                    video_detail["score"] = video_score
                    video_detail["correct_answers"] = correct_answers
                    video_detail["total_questions"] = total_questions
                    
                    total_score += video_score
                    valid_items += 1
                    
                    print(f"\nVideo evaluation completed:")
                    print(f"  - Correct answers: {correct_answers}/{total_questions}")
                    print(f"  - Score: {video_score:.4f}")
                else:
                    video_detail["score"] = 0.0
                    video_detail["correct_answers"] = 0
                    video_detail["total_questions"] = 0
                
                detailed_results.append(video_detail)
                if video_timing is not None:
                    self.timer.finish_video(video_timing)
        
        # Calculate average score
        average_score = total_score / valid_items if valid_items > 0 else 0.0
        
        print(f"\n===== Physics Reasoning Evaluation Summary =====")
        print(f"Total videos: {valid_items}")
        print(f"Average score: {average_score:.4f}")
        print(f"================================================")
        
        return average_score, detailed_results

    def retry_null_answers(self, results_data: Dict) -> Tuple[Dict, int]:
        """
        Scan results for questions with null answers and re-answer them using the API.
        Returns (updated_results_data, replaced_count).
        """
        # Collect all null entries: (v_idx, q_idx, question, caption)
        null_entries = []
        for v_idx, vd in enumerate(results_data.get('video_details', [])):
            for q_idx, qd in enumerate(vd.get('question_details', [])):
                if qd.get('answer') == 'null':
                    null_entries.append((v_idx, q_idx, qd['question'], qd['video_caption']))

        if not null_entries:
            print("No null answers found, nothing to retry.")
            return results_data, 0

        print(f"Found {len(null_entries)} null answers across {len(results_data.get('video_details', []))} videos, retrying...")

        # Re-answer concurrently with rate limiting and circuit breaker
        consecutive_failures = 0
        replaced = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_entry = {}
            for i, (v_idx, q_idx, question, caption) in enumerate(null_entries):
                if i > 0:
                    sleep(self.request_delay)
                future_to_entry[executor.submit(self.answer_physics_question, question, caption)] = (v_idx, q_idx)

            for future in as_completed(future_to_entry):
                v_idx, q_idx = future_to_entry[future]
                try:
                    answer = future.result()
                    consecutive_failures = 0
                except Exception as e:
                    print(f"Retry failed for video {v_idx}, question {q_idx}: {e}")
                    answer = 'null'
                    consecutive_failures += 1
                    if consecutive_failures >= self.max_consecutive_failures:
                        print(f"ERROR: {consecutive_failures} consecutive retry failures, aborting to avoid triggering risk controls")
                        for f in future_to_entry:
                            f.cancel()
                        raise RuntimeError(
                            f"Aborted during null retry: {consecutive_failures} consecutive API call failures. "
                            "Consider increasing PR_REQUEST_DELAY, reducing PR_MAX_WORKERS, or checking the API service."
                        )

                if answer != 'null':
                    replaced += 1
                vd = results_data['video_details'][v_idx]
                vd['question_details'][q_idx]['answer'] = answer
                vd['question_details'][q_idx]['is_correct'] = (answer == 'yes')
                vd['answers'][q_idx] = answer

        # Recalculate per-video scores
        for vd in results_data.get('video_details', []):
            answers = vd.get('answers', [])
            correct = sum(1 for a in answers if a == 'yes')
            total = len(answers)
            vd['correct_answers'] = correct
            vd['total_questions'] = total
            vd['score'] = correct / total if total > 0 else 0.0

        # Recalculate evaluation summary
        scores = [vd.get('score', 0.0) for vd in results_data.get('video_details', [])]
        results_data['evaluation_summary'] = {
            'total_videos': len(results_data.get('video_details', [])),
            'total_score': round(sum(scores), 1),
            'average_score': sum(scores) / len(scores) if scores else 0.0,
        }

        remaining = sum(1 for vd in results_data.get('video_details', [])
                        for qd in vd.get('question_details', [])
                        if qd.get('answer') == 'null')
        print(f"Retry round complete: replaced {replaced} nulls, {remaining} nulls remain")
        return results_data, replaced


def compute_physics_realism(json_dir: str, device: str, submodules_dict: Dict[str, Any], **kwargs) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Calculate physics reasoning score
    
    Args:
        json_dir: JSON file path
        device: Device
        submodules_dict: Submodules dictionary
        
    Returns:
        Tuple[float, List[Dict]]: (score, detailed results)
    """
    # Load data
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension='physics_realism', lang='en')

    # Resolve relative video paths using dataset base directory
    dataset_json = kwargs.get('dataset_json', '')
    dataset_base_dir = os.environ.get('DATASET_BASE_DIR', '')
    if not dataset_base_dir and dataset_json:
        # Infer base dir: dataset_json is like /base/condition_to_4D/.../file.json
        # and video paths start with condition_to_4D/...
        parts = dataset_json.split('/condition_to_4D/')
        if len(parts) > 1:
            dataset_base_dir = parts[0]
    if dataset_base_dir:
        for item in prompt_dict_ls:
            resolved = []
            for vp in item.get('video_list', []):
                if not os.path.isabs(vp) and not os.path.exists(vp):
                    full = os.path.join(dataset_base_dir, vp)
                    resolved.append(full)
                else:
                    resolved.append(vp)
            item['video_list'] = resolved
    
    # Ensure compatibility
    # if COMPATIBILITY_AVAILABLE:
    #     prompt_dict_ls = ensure_auxiliary_info_compatibility(prompt_dict_ls, 'physics_reasoning')
    
    timer = get_timer(submodules_dict, 'physics_realism', device)
    load_started = timer.start_model_load()
    from caption_method.VideoImg_caption import preload_keye_caption_model
    preload_keye_caption_model(os.environ.get("CAPTION_DEVICE", device))
    evaluator = PhysicsReasoningEvaluator(device, timer=timer)
    timer.finish_model_load(load_started)
    
    print("Starting physics reasoning evaluation...")
    #breakpoint()
    
    # Evaluate one dataset item at a time so a corrupt video or transient API
    # failure does not discard timings for the remaining videos. The Keye model
    # and API client above remain loaded once for the whole dimension.
    detailed_results = []
    for prompt_item in prompt_dict_ls:
        try:
            _, item_results = evaluator.evaluate_physics_reasoning([prompt_item])
        except Exception as exc:
            timer.fail_active(exc)
            print(
                f"[physics_realism] Failed item {prompt_item.get('video_list', [])}: "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        detailed_results.extend(item_results)
    scores = [float(item.get('score', 0.0)) for item in detailed_results]
    average_score = sum(scores) / len(scores) if scores else 0.0
    
    # Save detailed results
    try:
        output_dir = os.path.dirname(json_dir)
        dim_name = os.path.splitext(os.path.basename(json_dir))[0]
        model = kwargs.get('model', '')
        dataset_json = kwargs.get('dataset_json', '')
        dataset_base = os.path.splitext(os.path.basename(dataset_json))[0] if dataset_json else 'dataset'
        suffix = f"{dim_name}__{model}__{dataset_base}_results.json" if model else f"{dim_name}_results.json"
        output_file = os.path.join(output_dir, suffix)
        
        detailed_output = {
            "evaluation_summary": {
                "total_videos": len(detailed_results),
                "total_score": round(sum([r.get('score', 0) for r in detailed_results]),1),
                "average_score": average_score,
            },
            "video_details": detailed_results
        }

        # Auto-retry null answers (on by default, set PR_AUTO_RETRY_NULLS=0 to disable)
        if os.environ.get("PR_AUTO_RETRY_NULLS", "1") != "0":
            postprocess_started = timer.start_postprocess()
            max_rounds = int(os.environ.get("PR_MAX_NULL_RETRY_ROUNDS", "3"))
            for round_num in range(1, max_rounds + 1):
                detailed_output, replaced = evaluator.retry_null_answers(detailed_output)
                remaining = sum(
                    1 for vd in detailed_output.get('video_details', [])
                    for qd in vd.get('question_details', [])
                    if qd.get('answer') == 'null'
                )
                print(f"Auto-retry nulls round {round_num}/{max_rounds}: replaced={replaced}, remaining={remaining}")
                if remaining == 0:
                    print("All null answers resolved during auto-retry.")
                    break
                if replaced == 0:
                    print("No null answers were replaced this round, stopping auto-retry.")
                    break
            average_score = detailed_output['evaluation_summary']['average_score']
            detailed_results = detailed_output['video_details']
            timer.finish_postprocess(postprocess_started)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_output, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {output_file}")

    except Exception as e:
        print(f"Error saving JSON file: {str(e)}")

    return average_score, detailed_results


def retry_nulls_in_results_file(results_file: str, device: str, max_rounds: int = 3) -> Dict:
    """
    Load existing physics realism results, find null answers, and re-answer them.
    Runs up to max_rounds passes until no nulls remain or no progress is made.
    Saves updated results back to the same file.
    """
    with open(results_file, 'r', encoding='utf-8') as f:
        results_data = json.load(f)

    evaluator = PhysicsReasoningEvaluator(device)

    for round_num in range(1, max_rounds + 1):
        results_data, replaced = evaluator.retry_null_answers(results_data)
        remaining = sum(
            1 for vd in results_data.get('video_details', [])
            for qd in vd.get('question_details', [])
            if qd.get('answer') == 'null'
        )
        print(f"Null retry round {round_num}/{max_rounds}: replaced={replaced}, remaining={remaining}")
        if remaining == 0:
            print("All null answers resolved.")
            break
        if replaced == 0:
            print("No progress made in this round, stopping retries.")
            break

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    print(f"Updated results saved to: {results_file}")

    return results_data


# if __name__ == "__main__":
#     # Test case
#     test_json_dir = "dimension_description_json/physics_reasoning.json"
#     device = "cpu"
#     submodules_dict = {}
    
#     score, results = compute_physics_reasoning(test_json_dir, device, submodules_dict)
#     print(f"Test completed, score: {score:.4f}")
