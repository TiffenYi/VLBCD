import os
import cv2
import numpy as np
from tqdm import tqdm


def compare_masks_with_tp(gt_folder, pred_folder, output_folder):
    """
    对比GT和预测结果，生成误差可视化图：
    - 白色（TP）：预测正确部分
    - 红色（FP）：错检
    - 蓝色（FN）：漏检

    参数:
        gt_folder: Ground Truth文件夹路径
        pred_folder: 预测结果文件夹路径
        output_folder: 输出误差图的文件夹路径
    """
    os.makedirs(output_folder, exist_ok=True)
    gt_files = sorted(os.listdir(gt_folder))
    pred_files = sorted(os.listdir(pred_folder))

    for gt_file, pred_file in tqdm(zip(gt_files, pred_files), total=len(gt_files)):
        # 读取并二值化图片
        gt = cv2.imread(os.path.join(gt_folder, gt_file), cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(os.path.join(pred_folder, pred_file), cv2.IMREAD_GRAYSCALE)
        _, gt = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)
        _, pred = cv2.threshold(pred, 127, 255, cv2.THRESH_BINARY)

        # 计算FP、FN、TP
        fp = np.where((pred == 255) & (gt == 0), 255, 0).astype(np.uint8)  # 错检（红色）
        fn = np.where((gt == 255) & (pred == 0), 255, 0).astype(np.uint8)  # 漏检（蓝色）
        tp = np.where((gt == 255) & (pred == 255), 255, 0).astype(np.uint8)  # 正确检测（白色）

        # 创建彩色误差图（BGR格式）
        error_map = np.zeros((*gt.shape, 3), dtype=np.uint8)
        error_map[:, :, 2] = fp  # 红色通道（FP）
        error_map[:, :, 0] = fn  # 蓝色通道（FN）
        error_map[:, :, :] += np.stack([tp, tp, tp], axis=-1)  # 白色（TP覆盖所有通道）

        # 保存结果
        output_path = os.path.join(output_folder, gt_file)
        cv2.imwrite(output_path, error_map)

    print(f"可视化完成！结果已保存至: {output_folder}")


# 使用示例
if __name__ == "__main__":
    gt_folder = "C:/Users/admin/Desktop/gt_levir_256_test"  # 替换为GT文件夹路径
    pred_folder = "C:/Users/admin/Desktop/box2cd_clip_fpn_res50_levir_v5"  # 替换为预测结果文件夹路径
    output_folder = "C:/Users/admin/Desktop/result"  # 替换为输出文件夹路径

    compare_masks_with_tp(gt_folder, pred_folder, output_folder)

