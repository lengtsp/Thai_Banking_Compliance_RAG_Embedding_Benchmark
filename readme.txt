conda activate py131
cd "/home/indows-11/my_code/VENV"
source venv_qwen3vl/bin/activate

cd "/home/indows-11/my_code/claude code/compare_qwen3_embedding_0.6b_4b_8b_bgm-m3-"

python app.py

#### git
cd "/home/indows-11/my_code/claude code/embedding evalute"
sudo chown -R $USER .

git init
git branch -M main



git config --global user.email "thodsapon.th@gmail.com"
git config --global user.name "leng"