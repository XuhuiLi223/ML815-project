#!/bin/bash

# 完整的 Git 推送流程脚本
# 目标仓库: git@github.com:XuhuiLi223/ML815-project.git

set -e  # 遇到错误立即退出

echo "=== 开始 Git 推送流程 ==="

# 1. 检查当前状态
echo ""
echo "1. 检查当前 Git 状态..."
git status

# 2. 更新远程仓库地址
echo ""
echo "2. 更新远程仓库地址..."
git remote set-url origin git@github.com:XuhuiLi223/ML815-project.git
# 或者如果远程不存在，使用：
# git remote add origin git@github.com:XuhuiLi223/ML815-project.git

# 验证远程仓库
echo "当前远程仓库："
git remote -v

# 3. 添加所有文件（.gitignore 会自动排除不需要的文件）
echo ""
echo "3. 添加所有文件..."
git add .

# 4. 检查将要提交的文件
echo ""
echo "4. 将要提交的文件："
git status --short

# 5. 提交更改
echo ""
echo "5. 提交更改..."
read -p "请输入提交信息 (直接回车使用默认信息): " commit_msg
if [ -z "$commit_msg" ]; then
    commit_msg="Add multi-task support: classification, detection, segmentation, regression

- Refactored code structure with utils.py, argument.py, and task.py
- Added support for multiple tasks (classification, detection, segmentation, regression)
- Implemented metrics tracking (throughput, peak memory, MFU)
- Fixed DDP port conflicts and regression task dtype issues
- Added proper segmentation evaluation (mIoU, pixel accuracy)"
fi

git commit -m "$commit_msg"

# 6. 推送到远程仓库
echo ""
echo "6. 推送到远程仓库..."
echo "选择推送方式："
echo "  1) 推送到 main 分支 (git push -u origin main)"
echo "  2) 推送到 master 分支 (git push -u origin master)"
echo "  3) 强制推送 (git push -u origin main --force) - 谨慎使用！"
read -p "请选择 (1/2/3，默认1): " push_option

case $push_option in
    2)
        git push -u origin master
        ;;
    3)
        read -p "确认要强制推送吗？这将覆盖远程仓库的历史 (y/N): " confirm
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            git push -u origin main --force
        else
            echo "取消强制推送"
            exit 1
        fi
        ;;
    *)
        git push -u origin main
        ;;
esac

echo ""
echo "=== Git 推送完成！ ==="
echo "仓库地址: https://github.com/XuhuiLi223/ML815-project"

