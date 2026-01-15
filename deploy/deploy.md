# 克隆项目  git clone https://github.com/muzishen/IMAGDressing.git
cd /root
git clone https://github.com/your-username/IMAGDressing.git
cd IMAGDressing

# 创建新的虚拟环境
conda create -n IMAGDressing python=3.8 -y
conda activate IMAGDressing

# 按正确顺序安装
# 创建 constraints.txt 文件来锁定 PyTorch 版本
cat > constraints.txt << 'EOF'
torch==2.4.1+cu118
torchvision==0.19.1+cu118
torchaudio==2.4.1+cu118
EOF

# 先安装 PyTorch
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118

# 然后使用 constraints.txt 安装其他依赖
pip install -r requirements_no_torch.txt -c constraints.txt

pip install insightface
pip install onnxruntime

cd /root/autodl-tmp/IMAGDressing
source /etc/network_turbo
export HF_ENDPOINT=https://hf-mirror.com
export NO_ALBUMENTATIONS_UPDATE=1
pip install huggingface-hub -U

# 编辑 ~/.bashrc
echo 'export HF_HOME=/root/autodl-tmp/huggingface' >> ~/.bashrc
echo 'export TRANSFORMERS_CACHE=/root/autodl-tmp/huggingface' >> ~/.bashrc
source ~/.bashrc

huggingface-cli download --resume-download h94/IP-Adapter-FaceID \
  --local-dir ./ckpt/IP-Adapter-FaceID \
  --local-dir-use-symlinks False

cd /root/autodl-tmp/IMAGDressing
python inference_IMAGdressing_ipa_controlnetpose.py \
  --cloth_path cloth.jpg \
  --face_path face.jpg \
  --pose_path pose.jpg \
  --ip_ckpt /root/autodl-tmp/IMAGDressing/ckpt/IP-Adapter-FaceID/ip-adapter-faceid-plusv2_sd15.bin \
  --model_ckpt /root/autodl-tmp/IMAGDressing/ckpt/IMAGDressing-v1_512.pt \
  --output_path ./output_sd \
  --device cuda:0


=========================================================================================
使用界面：
pip install gradio>=3.50.0
pip install modelscope>=1.9.5
pip install controlnet-aux>=0.0.6
pip install opencv-python-headless>=4.8.0
# 指定一个较宽泛的版本，让pip自动协调依赖，可能避免冲突
pip install "tensorflow<2.14" -i https://mirrors.aliyun.com/pypi/simple/

pip uninstall onnxruntime -y
pip install onnxruntime-gpu==1.17.1
pip uninstall modelscope -y
pip install modelscope==1.9.2
pip install mmcv==1.7.2
pip install mmdet



# 完整示例（假设模型权重在当前目录）
python app.py --model_weight ./ckpt/IMAGDressing-v1_512.pt --server_port 7860





=========================================================================================
git clone https://github.com/muzishen/IMAGDressing.git
# 或使用国内镜像
# git clone https://gitee.com/empty-snow/IMAGDressing.git


AutoDL 镜像

    Python >= 3.11 (Recommend to use Anaconda or Miniconda)
    PyTorch >= 2.0.0
    cuda==12.2

conda create --name IMAGDressing python=3.11
conda activate IMAGDressing
pip install -U pip


# Install requirements
pip install -r requirements.txt


pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

验证：
python -c "
import torch
print(f'[✅] PyTorch版本: {torch.__version__}')
print(f'[✅] CUDA是否可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'[✅] GPU设备: {torch.cuda.get_device_name(0)}')

import numpy, diffusers, transformers
print(f'[✅] NumPy版本: {numpy.__version__}')
print('[✅] 核心AI库导入成功。')
"

# 示例命令格式，具体请以项目最新文档为准
python inference.py --garment_path ./cloth.jpg --face_path ./face.jpg --pose_path ./pose.jpg







======================================================================================
IMAGDressing 是一个虚拟穿衣工具，支持从线上便捷测试到本地灵活部署两种方式，具体对比见下表：

| 特性 | **在线测试 (推荐新手)** | **本地部署 (适合开发者)** |
| :--- | :--- | :--- |
| **使用方式** | 访问Hugging Face Spaces网页 | 在本地电脑或服务器上搭建环境并运行 |
| **核心条件** | **服装图**、**人脸图**、**姿势图**三要素 | 准备好输入图像，通过Python脚本或命令行运行 |
| **优点** | 无需安装，免费且无使用限制 | 完全离线，可深度定制，便于集成和二次开发 |
| **缺点** | 功能相对固定，依赖网络 | 环境配置复杂，对硬件（尤其是GPU）有一定要求 |
| **核心用途** | 快速体验和直观效果预览 | 研究、开发或需要批量处理的场景 |

### 🛠️ 在线快速测试
这是上手最快的方式，步骤如下：
1.  **访问平台**：打开 IMAGDressing 在 **Hugging Face Spaces** 上的平台。
2.  **准备三要素**：在页面左侧，分别上传或使用示例图片准备：
    *   **服装**：目标服装的清晰图片。
    *   **人脸**：希望出现在模特脸上的人脸图片。
    *   **姿势**：希望模特摆出的姿势参考图。
    *   对于“人脸”和“姿势”，记得勾选下方的 `Use face` 或 `Use pose` 选项，以告知模型使用你提供的图片。
3.  **生成与调整**：点击 **`Dressing`** 按钮开始生成。你还可以展开 **`Advanced Settings`** 调整高级参数（如去噪步数、引导尺度等）来优化效果。

### 💻 本地部署运行
如果你需要更多控制权，可以按照以下步骤在本地部署：
**1. 准备环境**
*   **系统**：推荐使用 **Ubuntu**，教程较为完整。
*   **Python**：需 **Python >= 3.8**，推荐使用 Anaconda 管理环境。
*   **CUDA**：为确保PyTorch能使用GPU加速，需要安装 **CUDA 11.8** 或 **12.1**。
*   **Git**：用于拉取代码。

**2. 安装步骤**
*   **克隆代码**：从GitHub或国内镜像（如Gitee）克隆项目仓库。
```bash
git clone https://github.com/muzishen/IMAGDressing.git
# 或使用国内镜像
# git clone https://gitee.com/empty-snow/IMAGDressing.git
```
*   **创建虚拟环境**：使用Conda创建一个独立环境。
```bash
conda create --name IMAGDressing python=3.8.10
conda activate IMAGDressing
```
*   **安装依赖**：进入项目目录，安装所需Python库。
```bash
cd IMAGDressing
pip install -U pip
# 可能需要根据实际情况微调requirements.txt中的版本
pip install -r requirements.txt
```
*   **下载预训练模型**：根据项目 `README` 说明，从Hugging Face或指定链接下载所需模型权重文件，并放置到正确的目录。

**3. 启动与测试**
*   具体启动命令需参考项目文档，通常是一个Python脚本，需要指定服装、人脸、姿势等输入图像的路径以及参数。
```bash
# 示例命令格式，具体请以项目最新文档为准
python inference.py --garment_path ./cloth.jpg --face_path ./face.jpg --pose_path ./pose.jpg
```

### ⚠️ 注意事项
*   **硬件要求**：本地部署对GPU显存有要求（建议8GB以上），否则可能无法运行或速度很慢。
*   **输入图片质量**：为了获得好的效果，请提供**背景干净、服装平整、人物姿态清晰**的图片作为输入。
*   **官方信息**：模型由研究团队开源，他们**未开发任何官方应用**（网页端或App），请以GitHub仓库内容为准。模型的输出存在一定随机性，且可能被输入误导。

总的来说，如果你是第一次接触，强烈建议从 **在线测试** 开始。如果想深入研究或整合到自己的项目中，再考虑 **本地部署**。

如果你能告诉我你的具体使用场景（例如，是个人体验，还是用于开发集成），我可以给你更具体的建议。