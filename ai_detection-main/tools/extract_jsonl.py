import os

def extract_lines_from_jsonl(input_filepath, output_filepath, num_lines_to_extract):
    """
    지정된 JSONL 파일에서 원하는 줄 수만큼만 읽어와 새로운 파일로 저장합니다.

    Args:
        input_filepath (str): 원본 JSONL 파일의 경로.
        output_filepath (str): 추출된 내용을 저장할 새 파일의 경로.
        num_lines_to_extract (int): 추출할 줄의 수.
    """
    try:
        # 출력 디렉토리가 존재하지 않으면 생성합니다.
        output_dir = os.path.dirname(output_filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"ℹ️ 출력 디렉토리 '{output_dir}'를 생성했습니다.")

        with open(input_filepath, 'r', encoding='utf-8') as infile, \
             open(output_filepath, 'w', encoding='utf-8') as outfile:
            for i, line in enumerate(infile):
                if i >= num_lines_to_extract:
                    break
                outfile.write(line)
        print(f"✅ {num_lines_to_extract}줄이 '{input_filepath}'에서 '{output_filepath}'(으)로 성공적으로 추출되었습니다.")
    except FileNotFoundError:
        # 이 부분이 중요합니다. input_filepath가 실제로 무엇이었는지 다시 출력하여 확인합니다.
        print(f"❌ 오류: '{input_filepath}' 파일을 찾을 수 없습니다. 경로를 확인해 주세요.")
        print(f"   입력하신 원본 파일 경로는 '{original_file_path_debug}' 였습니다.")
    except Exception as e:
        print(f"❌ 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    # 1. 원본 JSONL 파일 경로를 입력하세요.
    # 예시: original_data.jsonl 또는 /home/user/data/original_data.jsonl
    original_file_path = input("원본 JSONL 파일의 경로를 입력해 주세요: ")
    # 디버그를 위해 입력받은 원본 파일 경로를 전역 변수로 저장합니다.
    global original_file_path_debug
    original_file_path_debug = original_file_path
    print(f"--- 디버그 정보: 입력받은 원본 파일 경로: '{original_file_path_debug}' ---")


    # 2. 새로 만든 파일을 저장할 디렉토리 경로를 입력하세요.
    # 예시: extracted_files 또는 /home/user/output
    output_directory = input("새로 만든 파일을 저장할 디렉토리 경로를 입력해 주세요 (기본값: 현재 디렉토리): ")
    if not output_directory:
        output_directory = "." # 현재 디렉토리

    # 3. 새로 생성될 파일의 이름을 입력하세요.
    # 예시: extracted_data.jsonl
    new_file_name = input("새로 생성될 파일의 이름을 입력해 주세요 (예: extracted_data.jsonl): ")

    # 4. 추출할 줄 수를 입력하세요. (250000 줄이 기본값입니다.)
    while True:
        try:
            lines_to_extract_str = input("추출할 줄 수를 입력해 주세요 (기본값: 250000): ")
            if not lines_to_extract_str:
                lines_to_extract = 250000
            else:
                lines_to_extract = int(lines_to_extract_str)
            if lines_to_extract <= 0:
                print("줄 수는 0보다 커야 합니다. 다시 입력해 주세요.")
            else:
                break
        except ValueError:
            print("유효한 숫자를 입력해 주세요.")

    # 최종 출력 파일 경로를 조합합니다.
    final_output_filepath = os.path.join(output_directory, new_file_name)

    extract_lines_from_jsonl(original_file_path, final_output_filepath, lines_to_extract)
    print("프로그램이 종료되었습니다.")