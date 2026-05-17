# 사용법: 'logs'라는 폴더를 만들고 그 안에 .txt 로그들을 넣은 뒤 실행하세요.
# plot_multiple_logs('./logs'

import matplotlib.pyplot as plt
import re
import os

def plot_multiple_logs(log_folder):
    plt.figure(figsize=(12, 7))
    
    # 폴더 내의 모든 파일 목록 가져오기
    log_files = [f for f in os.listdir(log_folder) if f.endswith('.txt')]
    
    if not log_files:
        print("로그 파일을 찾을 수 없습니다. 경로를 확인하세요.")
        return

    # 각 파일마다 다른 색상과 스타일을 적용하기 위함
    for i, file_name in enumerate(log_files):
        iters = []
        test_acc = []
        
        file_path = os.path.join(log_folder, file_name)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            log_data = f.read()
            
        # 정규표현식으로 데이터 추출
        pattern = r"iter: \((\d+):\d+\) train acc:  [\d.]+  test acc:  ([\d.]+)"
        matches = re.findall(pattern, log_data)
        
        for m in matches:
            iters.append(int(m[0]))
            test_acc.append(float(m[1]))
            
        # 그래프에 선 추가 (파일명을 라벨로 사용)
        plt.plot(iters, test_acc, label=f'Test: {file_name}', linewidth=2)

    # 그래프 설정
    plt.axhline(y=0.66, color='red', linestyle='--', label='Target (66%)') # 목표선
    plt.title('CNN Model Optimization Comparison', fontsize=16)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.0)
    
    plt.savefig('comparison_result.png', dpi=300)
    plt.show()
