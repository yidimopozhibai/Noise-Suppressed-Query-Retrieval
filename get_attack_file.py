import os
import shutil
from glob import glob
from tqdm import tqdm
from get_one_map import rtdetr_heatmap, get_params

# 路径配置
IMG_ROOT = 'dataset/images'
LBL_ROOT = 'dataset/labels'
# SUBSETS  = ['train', 'val', 'test']
SUBSETS  = ['val']
SUF      = '_attack'

# 静默模式参数（不弹窗、不画框）
params = get_params()
params['show_box'] = False          # 不画框
params['renormalize'] = False       # 不限制热力图在框内
model = rtdetr_heatmap(**params)

def attack_subset(sub: str):
    src_img = os.path.join(IMG_ROOT, sub)
    dst_img = os.path.join(IMG_ROOT, sub + SUF)
    src_lbl = os.path.join(LBL_ROOT, sub)
    dst_lbl = os.path.join(LBL_ROOT, sub + SUF)

    os.makedirs(dst_img, exist_ok=True)
    if os.path.exists(src_lbl):
        shutil.copytree(src_lbl, dst_lbl, dirs_exist_ok=True)

    for img_path in tqdm(glob(os.path.join(src_img, '*')), desc=f'Attack {sub}'):
        if not img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_name = os.path.basename(img_path)

        # 执行攻击（内部已保存 GanAttack.jpg）
        model(img_path)

        # 复制攻击结果并保持原名
        shutil.copy('headmap/GanAttack.jpg', os.path.join(dst_img, img_name))

if __name__ == '__main__':
    for sub in SUBSETS:
        attack_subset(sub)
    print('✅ 攻击数据集生成完成！')