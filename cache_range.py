# prompt
# cache： S， E， B
# B显示cache能占用多少空间（行长度），S则为cache能取的颜色数量。
# 无论E多少，同一个内存地址都是那个cache块，只不过E>1时相同位置映射不等于一定冲突。

# 绘制两个矩阵，A左侧B右侧，矩阵大小为A是NxM，B是MxN
# 但在存储上，他们要得是三维张量，第三维度是一个数值表示颜色（不要用三元组）
# 绘制NxM和MxN个小格子，格子大小统一配置。
# A的起始地址看作是0计算A的每一个元素的地址（每个格子先假设为int，也就是4B），然后计算它属于哪一个cache组，随后将其设为cache组的颜色。
# B的起始地址为 A+一个偏移，偏移需要指定。默认指定为0x40000. 然后，类似A的计算，计算B的每一个元素地址，然后计算它属于哪一个cache组，随后将其设为cache组的颜色。

# cache组的颜色通过计算得来，从色彩空间中取8个色系列（红橙黄绿青蓝紫灰），每个颜色取8个渐变梯度，一共64种；cache组数不许超过64，否则报错。
# 颜色的映射是这样：类似cache中位映射的原理，相邻组取不同的色系的同一梯度，而不是相同色系的不同梯度。前一种人眼好区分，后一种人眼不好区分，且在仅仅用了部分组的情况下，颜色少得多，单调乏味。

# cache参数，N，M来自输入。输入 s（log2（S）），E（E），b（log2（B）），N，M。
# 要在N，M不相等的情况下仍然可以 工作。


"""
python cache_range.py  5 1 5 32 32

python cache_range.py  5 1 5 32 32 --offset 0x40000 --element_size 4
python cache_range.py  5 1 5 64 64 --offset 0x40000 --element_size 4
python cache_range.py  5 1 5 61 67 --offset 0x40000 --element_size 4
"""

import argparse

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

tick_gap = 4

def calculate_cache_set(address, block_size, num_sets):
    """计算内存地址对应的cache组号"""
    # The block number in memory
    block_number = address // block_size
    # The set index is the block number modulo the number of sets
    return block_number % num_sets


def generate_cache_colors(num_cache_sets):
    """生成64色自定义colormap，相邻组不同色系同梯度"""
    if num_cache_sets > 64:
        # This error check is also in the main visualization function, but good to have here too.
        print(
            f"Warning: Number of cache sets ({num_cache_sets}) exceeds the recommended maximum (64) for distinct colors."
        )
        print(
            "The color palette is designed for up to 64 distinct colors based on 8 families and 8 gradients."
        )
    # 8 color families, 8 gradients per family = 64 unique colors.
    # Colors are ordered such that:
    # Set 0: Family 0, Gradient 0
    # Set 1: Family 1, Gradient 0
    # ...
    # Set 7: Family 7, Gradient 0
    # Set 8: Family 0, Gradient 1
    # ... and so on.
    base_rgb_colors_0_255 = [
        (245, 34, 45),   # #f5222d
        (250, 140, 22),  # #fa8c16
        (250, 219, 20),  # #fadb14
        (82, 196, 26),   # #52c41a
        (19, 194, 194),  # #13c2c2
        (24, 144, 255),  # #1890ff
        (114, 46, 209),  # #722ed1
        (89, 89, 89),    # #595959
    ]

    num_color_families = 8
    gradients_per_family = 8  # Fixed at 8 to make up 64 total potential colors

    full_palette_rgb_0_1 = []  # List to store (R,G,B) tuples normalized to 0-1
    for grad_idx in range(gradients_per_family):  # Gradient level (0 to 7)
        for fam_idx in range(num_color_families):  # Color family index (0 to 7)
            base_r, base_g, base_b = base_rgb_colors_0_255[fam_idx]

            # Interpolate towards white to create gradients.
            # current_lightness_factor: 0 for base color, up to ~0.6 for lightest shade.
            # grad_idx = 0 results in the base color.
            # grad_idx = (gradients_per_family - 1) results in the lightest shade.
            if gradients_per_family > 1:
                # Max factor to add (to avoid pure white and keep distinctiveness)
                max_lightness_addition_factor = 0.9  # @ 越大颜色间差距越大
                current_lightness_factor = (
                    grad_idx / (gradients_per_family - 1)
                ) * max_lightness_addition_factor
            else:  # Only one gradient level defined
                current_lightness_factor = 0

            # Calculate the gradient color components
            r = base_r + (255 - base_r) * current_lightness_factor
            g = base_g + (255 - base_g) * current_lightness_factor
            b = base_b + (255 - base_b) * current_lightness_factor

            # Normalize to 0-1 range and clamp values
            final_r = min(max(r / 255.0, 0.0), 1.0)
            final_g = min(max(g / 255.0, 0.0), 1.0)
            final_b = min(max(b / 255.0, 0.0), 1.0)
            full_palette_rgb_0_1.append((final_r, final_g, final_b))

    # Return only the necessary number of colors
    return full_palette_rgb_0_1[:num_cache_sets]


def visualize_cache_mapping(s, E, b, N, M, offset_B=0x40000, element_size_bytes=4):
    """
    Visualizes cache set mapping for two matrices A and B.

    Args:
        s_log_sets (int): log2(S), where S is the number of cache sets.
        E_associativity (int): Cache associativity (E). (Not directly used in coloring by set index for this viz)
        b_log_block_size (int): log2(B), where B is the cache block size in bytes.
        N_rows_A (int): Number of rows for Matrix A (and columns for Matrix B).
        M_cols_A (int): Number of columns for Matrix A (and rows for Matrix B).
        offset_B (int, optional): Starting offset for Matrix B relative to the end of Matrix A. Defaults to 0x40000.
        element_size_bytes (int, optional): Size of each matrix element in bytes. Defaults to 4.
    """
    # 参数验证
    num_cache_sets = 2**s
    block_size_bytes = 2**b
    if num_cache_sets <= 0:
        print(f"Error: Number of cache sets ({num_cache_sets}) must be positive.")
        return
    if block_size_bytes <= 0:
        print(f"Error: Cache block size ({block_size_bytes}) must be positive.")
        return
    if num_cache_sets > 64:
        print(f"Error: Number of cache sets ({num_cache_sets}) exceeds the maximum supported (64).")
        print(
            "The color palette is designed for up to 64 distinct colors based on 8 families and 8 gradients."
        )
        return

    # 计算矩阵每个位置的缓存组号
    a_matrix = np.zeros((N, M), dtype=int)
    # Matrix A starts at address 0
    for r in range(N):
        for c in range(M):
            # Address of element A[r, c] assuming row-major layout
            addr = (r * M + c) * element_size_bytes
            set_idx = calculate_cache_set(addr, block_size_bytes, num_cache_sets)
            a_matrix[r, c] = set_idx

    b_matrix = np.zeros((M, N), dtype=int)
    for r_b in range(M):
        for c_b in range(N):
            addr = offset_B + (r_b * N + c_b) * element_size_bytes  # 固定偏移地址
            set_idx = calculate_cache_set(addr, block_size_bytes, num_cache_sets)
            b_matrix[r_b, c_b] = set_idx
    
    # 绘制图像
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), dpi=150)  # Adjust figsize as needed
    
    # 生成颜色映射
    # Define normalization for imshow: maps data values [0, num_cache_sets-1] to cmap
    # Handle num_cache_sets=1 case where max_val would be 0.
    active_palette = generate_cache_colors(num_cache_sets)
    cmap = mcolors.ListedColormap(active_palette)
    max_val_for_norm = max(0, num_cache_sets - 1)
    norm = mcolors.Normalize(vmin=0, vmax=max_val_for_norm)
    
    # Plot Matrix A
    axes[0].imshow(a_matrix, cmap=cmap, norm=norm, interpolation='nearest')
    axes[0].set_title(f"Matrix A ({N}x{M}) Cache Sets")
    axes[0].set_xlabel(f"Columns ({M})")
    axes[0].set_ylabel(f"Rows ({N})")

    # 设置A矩阵的Y轴刻度，显示所有0到N_rows_A-1的值，并旋转标签
    axes[0].set_yticks(np.arange(0, N, tick_gap))
    axes[0].set_yticklabels(np.arange(0, N, tick_gap), rotation=45, ha='right')  # ha='right' 使标签右对齐，避免重叠
    # 设置A矩阵的X轴刻度，显示所有0到M_cols_A-1的值，并旋转标签
    axes[0].set_xticks(np.arange(0, M, tick_gap))
    axes[0].set_xticklabels(np.arange(0, M, tick_gap), rotation=45, ha='right')  # ha='right' 使标签右对齐，避免重叠

    # Plot Matrix B
    axes[1].imshow(b_matrix, cmap=cmap, norm=norm, interpolation='nearest')
    axes[1].set_title(f"Matrix B ({M}x{N}) Cache Sets")
    axes[1].set_xlabel(f"Columns ({N})")
    axes[1].set_ylabel(f"Rows ({M})")

    # 设置B矩阵的Y轴刻度，显示所有0到M-1的值，并旋转标签
    axes[1].set_yticks(np.arange(0, M, tick_gap))
    axes[1].set_yticklabels(np.arange(0, M, tick_gap), rotation=45, ha='right')
    # 设置B矩阵的X轴刻度，显示所有0到N-1的值，并旋转标签
    axes[1].set_xticks(np.arange(0, N, tick_gap))
    axes[1].set_xticklabels(np.arange(0, N, tick_gap), rotation=45, ha='right')

    # Plot a shared colorbar
    # Create a ScalarMappable for the colorbar using the same norm and cmap
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])  # An empty array is fine as norm and cmap are already set
    # Adjust layout to make space for the colorbar on the right
    fig.subplots_adjust(right=0.85)
    # Define position for colorbar axes: [left, bottom, width, height] in figure coordinates
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])

    # Determine ticks for the colorbar
    cbar_ticks = np.arange(num_cache_sets)  # Show all set indices if few
    if num_cache_sets > 16:  # If many sets, reduce tick count for clarity
        cbar_ticks = np.linspace(0, num_cache_sets - 1, min(num_cache_sets, 17), dtype=int)
        # Ensure the last set index is included if not already present and num_cache_sets > 1
        if num_cache_sets > 1 and (num_cache_sets - 1) not in cbar_ticks:
            cbar_ticks = np.unique(np.append(cbar_ticks, num_cache_sets - 1))

    cbar = fig.colorbar(mappable, cax=cbar_ax, ticks=cbar_ticks)
    cbar.set_label('Cache Set Index')

    plt.suptitle(
        f"Cache Mapping (Sets: {num_cache_sets}, E: {E}, Block Size: {block_size_bytes}B)",
        fontsize=16,
    )
    # Adjust top/bottom margins for suptitle and labels
    fig.subplots_adjust(top=0.90, bottom=0.1)

    plt.show()

def init_argparse():
    parser = argparse.ArgumentParser(
        description=f"Visualize cache mapping for matrices A and B. Each element is colored based on the cache set it maps to.\n Example: python cache_range.py  5 1 5 32 32 --offset 0x40000 --element_size 4",
        formatter_class=argparse.RawTextHelpFormatter,  # For better help text formatting
    )
    parser.add_argument("s", type=int, 
        help="log2(S), where S is the number of cache sets (e.g., 3 for S=8 sets).")
    parser.add_argument("E", type=int,
        help="Cache associativity (E) (e.g., 1 for direct-mapped, 4 for 4-way set associative).",)
    parser.add_argument("b", type=int,
        help="log2(B), where B is the cache block size in bytes (e.g., 4 for B=16 bytes).",)
    parser.add_argument("N", type=int, 
        help="Number of rows for Matrix A (and columns for Matrix B).")
    parser.add_argument("M", type=int, 
        help="Number of columns for Matrix A (and rows for Matrix B).")
    parser.add_argument("--offset", type=str, default="0x40000",
        help="Starting offset for Matrix B (hex string, e.g., '0x40000'). Defaults to 0x40000 (262144 bytes).",)
    parser.add_argument("--element_size", type=int, default=4,
        help="Size of each matrix element in bytes (e.g., 4 for int). Defaults to 4.",)
    return parser

# --- Entry point for command-line execution ---
if __name__ == '__main__':
    args = init_argparse().parse_args()
    try:
        # Convert hex offset string to integer
        offset_val = int(args.offset, 16)
    except ValueError:
        print(
            f"Error: Invalid hexadecimal format for offset: '{args.offset}'. Please use '0x...' format."
        )
        # Fallback to default or exit, here we'll use default and notify.
        print("Using default offset 0x40000.")
        offset_val = 0x40000

    print(f"\nRunning cache visualization with parameters:")
    print(f"  s (log2(Sets)): {args.s} => {2**args.s} Sets")
    print(f"  E (Associativity): {args.E}")
    print(f"  b (log2(Block Size)): {args.b} => {2**args.b} Bytes/Block")
    print(f"  N (Matrix A rows, B cols): {args.N}")
    print(f"  M (Matrix A cols, B rows): {args.M}")
    print(f"  Offset for B: 0x{offset_val:X} ({offset_val} bytes)")
    print(f"  Element Size: {args.element_size} bytes")
    print("-" * 30)

    visualize_cache_mapping(
        args.s,
        args.E,
        args.b,
        args.N,
        args.M,
        offset_B=offset_val,
        element_size_bytes=args.element_size,
    )