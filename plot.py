"""
Usage example: 
$ python3 plot.py -s 32 -f ./32x32.trace

-s: Size of the matrix (must be a square matrix).
-f: Path to the trace file.
The trace file is generated by csapp's cachelab test-trans and csim-ref programs.
The generation process is as follows:

$ ./test-trans -M 32 -N 32
# pick your function i
$ ./csim-ref -v -s 5 -E 1 -b 5 -t ./trace.fi > ./32x32.trace

The `base_address` and `threshold_address` will be calculated automatically.
base_address 和 threshold_address 会自动计算。
运行后，按 p 键暂停，再按 p 键继续，按 ESC 键退出。
暂不支持非正方形矩阵。
"""
import cv2
import numpy as np
import os
import argparse

# 全局变量
matrix_size = 32  # This should be specified correctly by the user
base_address = 0x3fffffff
threshold_address = 0x3fffffff
save_images = True  # Set this to True to save images, False to just display
file_path = f'../traces/32x32.trace'


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize matrix operations from trace files.")
    parser.add_argument("-s", "--size", type=int, required=True,
                        help="Size of the matrix (must be a square matrix).")
    parser.add_argument("-st", "--save-images", action='store_true',
                        help="Save images to disk instead of just displaying them.", default=False)
    parser.add_argument("-f", "--file-path", type=str, 
                        required=True,
                        help="Path to the trace file.")
    return parser.parse_args()


def parse_address(address):
    current_address = int(address, 16)
    # print(current_address, base_address, threshold_address)
    if current_address >= threshold_address:
        return 'B', current_address - threshold_address  # B matrix operation
    elif current_address >= base_address:
        return 'A', current_address - base_address  # A matrix operation
    return None, None

def update_frame(matrix, offset, result):
    row = (offset // 4) // matrix_size
    col = (offset // 4) % matrix_size
    # 边界检查
    if row >= matrix_size or col >= matrix_size:
        print("offset:", offset)
        print(f"Index out of bounds: row={row}, col={col}, matrix_size={matrix_size}")
        exit(1)
    color_hit = (0x1b, 0x44, 0x00)  # RGB for #00441b
    color_miss = (0xf5, 0xfc, 0xf7)  # RGB for #f7fcf5
    matrix[row, col] = color_hit if result == 'h' else color_miss
    return row, col

def create_matrix(matrix_size):
    return np.zeros((matrix_size, matrix_size, 3), dtype=np.uint8)  # 3 for RGB channels


def display_matrices(a_matrix, b_matrix, current_row, current_col, operation, scale=20, frame_count=0):
    a_image = cv2.resize(a_matrix, (matrix_size * scale, matrix_size * scale), interpolation=cv2.INTER_NEAREST)
    b_image = cv2.resize(b_matrix, (matrix_size * scale, matrix_size * scale), interpolation=cv2.INTER_NEAREST)
    combined_image = np.hstack((a_image, b_image))
    offset = a_image.shape[1] if operation == 'B' else 0
    cv2.rectangle(combined_image, (current_col * scale + offset, current_row * scale),
                  ((current_col + 1) * scale + offset, (current_row + 1) * scale), (0x42, 0x42, 0xdb), 3)  # Red border
    if save_images:
        output_path = f'/tmp/pic_{matrix_size}x{matrix_size}_cache_footprint_{file_path.split("/")[-1].split(".")[0]}'
        # print(output_path)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        cv2.imwrite(os.path.join(output_path, f'frame_{frame_count:04d}.png'), combined_image)
    else:
        cv2.imshow('Cache Matrix', combined_image)                
        key = cv2.waitKey(3)   # @  change the speed of the display
        if key == 27:
            return False
        if key == ord('p'):
            while True:
                key = cv2.waitKey(0)
                if key == ord('p'):
                    break
    return True

def main():
    global base_address, threshold_address  # 声明全局变量
    a_matrix = create_matrix(matrix_size)
    b_matrix = create_matrix(matrix_size)

    with open(file_path) as f:
        lines = [line.strip() for line in f if line.strip()][4:-2]
        partss = [line.split() for line in lines]
    
        # 检查以L开头的最小地址为base_address
        for parts in partss:
            if parts[0] == 'L':
                address = parts[1].split(',')[0]
                current_address = int(address, 16)
                if current_address < base_address:
                    base_address = current_address
        print(f"base_address: {base_address:x}")
        # 检查以S开头的最小地址为threshold_address
        for parts in partss:
            if parts[0] == 'S':
                address = parts[1].split(',')[0]
                current_address = int(address, 16)
                if current_address < threshold_address:
                    threshold_address = current_address
        print(f"threshold_address: {threshold_address:x}") 
        
        frame_count = 0
        for line in lines:
            parts = line.split()
            address = parts[1].split(',')[0]
            result = 'h' if 'hit' in line else 'm'
            operation_matrix, offset = parse_address(address)
            if operation_matrix:
                matrix_to_update = a_matrix if operation_matrix == 'A' else b_matrix
                current_row, current_col = update_frame(matrix_to_update, offset, result)
                if not display_matrices(a_matrix, b_matrix, current_row, current_col, operation_matrix, frame_count=frame_count):
                    break
                frame_count += 1
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    args = parse_args()  # 解析命令行参数
    matrix_size = args.size
    save_images = args.save_images
    file_path = args.file_path
    main()