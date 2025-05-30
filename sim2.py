"""
使用示例：
$ python3 sim2.py -r 32 -c 32 -f ./32x32.trace

-r/--rows: 矩阵行数
-c/--cols: 矩阵列数
-f: 跟踪文件路径
-S/--save-images: 保存图像到磁盘
-n/--no-display: 关闭显示
-F/--fast: 快速模式（禁用暂停控制）

trace文件由csapp的cachelab test-trans和csim-ref程序生成。
示例：
$ ./test-trans -M 32 -N 32
$ ./csim-ref -v -s 5 -E 1 -b 5 -t ./trace.fi > ./32x32.trace

base_address 和 threshold_address 会自动计算。
对 .trace 文件的截取和 csim-ref 的输出格式耦合。 cim-ref首先会输出4行无用的trace，随后是转置的trace，最后一行输出hit/miss/eviction情况。
运行后，按 p 键暂停，再按 p 键继续，按 ESC 键退出。
暂不支持非正方形矩阵。
"""

import cv2
import numpy as np
import os
import argparse

# 配色：opencv BGR 格式
COLOR_HIT = (0x1B, 0x44, 0x00)  # RGB for #00441b
COLOR_MISS = (0xF5, 0xFC, 0xF7)  # RGB for #f7fcf5
COLOR_CURRENT = (0x42, 0x42, 0xDB)
COLOR_M_BORDER = (5, 176, 250)

# 缩放比例，作为常量
SCALE = 15


def create_image(rows, cols):
  """创建一个指定行和列的黑色图像（矩阵），用于可视化。"""
  return np.zeros((rows, cols, 3), dtype=np.uint8)  # 3 for BGR channels


def parse_args() -> argparse.Namespace:
  """解析命令行参数"""
  parser = argparse.ArgumentParser(description="矩阵操作缓存可视化工具")
  parser.add_argument("-r", "--rows", type=int, required=True, help="矩阵行数")
  parser.add_argument("-c", "--cols", type=int, required=True, help="矩阵列数")
  parser.add_argument("-f", "--file-path", type=str, required=True, help="跟踪文件路径")
  parser.add_argument("-S", "--save-images", action="store_true", help="保存图像到磁盘", default=False)
  parser.add_argument("-n", "--no-display", action="store_true", help="关闭显示窗口", default=False)
  parser.add_argument("-F", "--fast", action="store_true", help="快速模式（禁用暂停控制）", default=False,)
  return parser.parse_args()


def parse_address(addr: str, a_base_addr: int, b_base_addr: int):
  """根据地址判断是哪个矩阵，返回操作的矩阵 ('A' 或 'B') 和偏移量"""
  current_addr = int(addr, 16)
  matrix_type, addr_offset = None, None
  if current_addr >= b_base_addr:
    matrix_type = "B"
    addr_offset = current_addr - b_base_addr
  elif current_addr >= a_base_addr:
    matrix_type = "A"
    addr_offset = current_addr - a_base_addr

  return matrix_type, addr_offset

def parse_base_addr(traces: list):
  """从 trace 中解析 A 和 B 矩阵的基地址"""
  a_base_addr = 0x3FFFFFFF
  b_base_addr = 0x3FFFFFFF

  # 检查以L开头的最小地址为 A_BASE_ADDR
  for trace in traces:
    if trace[0] == "L":
      addr = trace[1].split(",")[0]
      current_address = int(addr, 16)
      if current_address < a_base_addr:
        a_base_addr = current_address

  # 检查以S开头的最小地址为 B_BASE_ADDR
  for trace in traces:
    if trace[0] == "S":
      addr = trace[1].split(",")[0]
      current_address = int(addr, 16)
      if current_address < b_base_addr:
        b_base_addr = current_address

  print(f"A base address: 0x{a_base_addr:x}")
  print(f"B base address: 0x{b_base_addr:x}")

  return a_base_addr, b_base_addr


def update_matrix(image, row, col, result):
  """更新矩阵图像的指定单元格颜色"""
  # 仅当 result 为 'h' (hit) 且当前单元格为黑色时，才更新颜色为 COLOR_HIT
  if result == "h":
    if np.array_equal(image[row, col], (0, 0, 0)):
      image[row, col] = COLOR_HIT
    # 此时当前单元格不是黑色，说明已经为miss，不可以更改为hit
  else:  # result is 'm' (miss)
    image[row, col] = COLOR_MISS
  return

def draw_combined_image(
    a_matrix: np.ndarray,
    b_matrix: np.ndarray,
    cur_pos: tuple[int, int],
    operation: str,
) -> np.ndarray:
    """生成合并后的可视化图像"""
    # 调整矩阵大小
    a_scaled = cv2.resize(
        a_matrix, 
        (a_matrix.shape[1] * SCALE, a_matrix.shape[0] * SCALE),
        interpolation=cv2.INTER_NEAREST
    )
    b_scaled = cv2.resize(
        b_matrix,
        (b_matrix.shape[1] * SCALE, b_matrix.shape[0] * SCALE),
        interpolation=cv2.INTER_NEAREST
    )
    combined = np.hstack((a_scaled, b_scaled))  # 水平拼接图像
    # 绘制矩阵边框
    # 矩阵A边框（左侧矩阵）
    cv2.rectangle(
        combined,
        (0, 0),
        (a_scaled.shape[1], a_scaled.shape[0]),  # 右下角坐标
        COLOR_M_BORDER,
        thickness=2
    )
     # 矩阵B边框（右侧矩阵）
    cv2.rectangle(
        combined,
        (a_scaled.shape[1], 0),  # 左上角x坐标为A矩阵宽度
        (combined.shape[1], b_scaled.shape[0]),  # 右下角坐标
        COLOR_M_BORDER,
        thickness=2
    )

    # 绘制当前操作高亮框（在边框上层）
    # 计算当前操作矩阵的X偏移量
    cur_row, cur_col = cur_pos
    x_offset = a_matrix.shape[1] * SCALE if operation == "B" else 0
    cv2.rectangle(
        combined,
        (x_offset + cur_col * SCALE, cur_row * SCALE),
        (x_offset + (cur_col + 1) * SCALE, (cur_row + 1) * SCALE),
        COLOR_CURRENT,
        3,
    )
    return combined

def display_frame(
    image: np.ndarray,
    fast_mode: bool,
) -> bool:
    """显示帧并处理用户输入"""
    cv2.imshow("Cache Matrix", image)
    if fast_mode:
        return cv2.waitKey(1) != 27  # ESC 键检测
    
    key = cv2.waitKey(350)
    if key == 27:  # ESC
        return False
    if key == ord("p"):  # 暂停控制
        while cv2.waitKey(0) not in (27, ord("p")):
            pass
    return True

def save_frame(
    image: np.ndarray,
    output_dir: str,
    frame_count: int,
) -> None:
    """保存帧到文件"""
    cv2.imwrite(os.path.join(output_dir, f"frame_{frame_count:04d}.png"), image)
    # \r 为回车，将光标移动到行首；end="" 阻止换行，这两个合起来让输出只在一行；
    # flush=True 刷新缓冲区，确保内容即时显示，避免延迟
    print(f"\rSaved frame {frame_count}", end="", flush=True)

def main():
  """
  主函数，处理命令行参数，读取 trace 文件，并进行矩阵可视化。
  """
  args = parse_args()  # 解析命令行参数

  # 使用解析到的参数初始化矩阵
  A = create_image(args.rows, args.cols)  # shape=(ROWS, COLS, 3) 三维张量
  B = create_image(args.cols, args.rows)  # B 矩阵通常是 A 的转置，所以行列互换
  
  # 构建输出路径
  output_dir = f'/tmp/{args.rows}x{args.cols}-cache_footprint-{os.path.basename(args.file_path).split(".")[0]}'
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  if args.save_images:
    print(f"Output dir: {output_dir}")
  
  with open(args.file_path) as f:
    # 截取 trace 文件内容，去除开头4行和末尾2行
    lines = [line.strip() for line in f if line.strip()][4:-2]
    # 将每行解析为元组，例如 [('L', '0xdeedbeef', 'miss'), ...]
    traces = [line.split() for line in lines]
    a_base_addr, b_base_addr = parse_base_addr(traces) # 解析基地址
    frame_count = 1
    for line in lines:
      trace = line.split()
      addr = trace[1].split(",")[0]
      result = "h" if "hit" in line else "m"
      # L后的为A，S后的为B
      matrix_type, addr_offset = parse_address(addr, a_base_addr, b_base_addr)
      if not matrix_type:
        print(f"Warning: \"{trace}\" parsing failed.")
        continue
      tar_image = A if matrix_type == "A" else B  # 选择目标矩阵
      cur_row = (addr_offset // 4) // tar_image.shape[1]
      cur_col = (addr_offset // 4) % tar_image.shape[1]
      update_matrix(tar_image, cur_row, cur_col, result)  # 更新矩阵
      combined_img = draw_combined_image(A, B, (cur_row, cur_col), matrix_type) # 绘制图像
      
      # 显示/保存处理
      if not args.no_display and not args.save_images:
        if not display_frame(combined_img, args.fast):
            break
      if args.save_images:
          save_frame(combined_img, output_dir, frame_count)
  
      frame_count += 1

  # 等待用户按任意键退出，然后关闭所有 OpenCV 窗口
  if not args.no_display:
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
  main()
