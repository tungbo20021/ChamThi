# Theo dõi các file lớn hơn 100MB
git lfs track "*.pt"
git lfs track "*.pth"
git lfs track "*.h5"
git lfs track "*.onnx"
git lfs track "*.weights"

# Thêm file .gitattributes vào repository
git add .gitattributes
git commit -m "Configure Git LFS tracking"