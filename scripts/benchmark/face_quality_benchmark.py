"""
from pathlib import Path
from scripts.benchmark.util import frames_to_evaluate_from_benchmark_dir, get_model_iteration_reached, \
    write_bench_generic, clean_output_csv, path_to_b64

import json
from utils import *
from utils.utils import *


def launch_benchmark_quality_score(benchmark_dir: Path, csv_file_name: str):
    benchmark_csv = clean_output_csv(benchmark_dir, csv_file_name)
    write_bench_generic(benchmark_csv, ["Model", "Target_iteration", "Quality_score"])
    for _, frame_path in frames_to_evaluate_from_benchmark_dir(benchmark_dir):
        model, iteration = get_model_iteration_reached(frame_path)
        quality_score = get_score_from_fqa_service(path_to_b64(frame_path))
        write_bench_generic(benchmark_csv, [model, str(iteration), str(quality_score)])


def get_score_from_fqa_service(reference_image_base64_str):
    url_image_quality_assessment = "http://localhost" + ":81"
    request_image_quality = {"Face_Image": reference_image_base64_str}
    input_json_image_quality = json.dumps(request_image_quality, indent=0)
    url_image_quality_api = url_image_quality_assessment + "/Face_Image_Quality_Assessment"
    response_image_quality_assessment = send_http_request_image_quality_assessment(url_image_quality_api,
                                                                                   input_json_image_quality)
    quality_assessment_result = response_image_quality_assessment.json()
    score_quality = quality_assessment_result["Overall_Quality_Score"]
    return score_quality


def send_http_request_image_quality_assessment(http_url, input_json_data):
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    session.headers = {
        'Content-Type': 'application/json',
        'Cache-Control': 'no-cache',
    }
    url = http_url
    response = session.post(url, data=input_json_data)

    if response.status_code == 200:
        str_logging_information = "successfully assessed image quality."
        print(str_logging_information)
    return response


if __name__ == '__main__':
    benchmark_dir = Path("")
    launch_benchmark_quality_score(benchmark_dir, "benchmark_quality_score.csv")
"""
