# Version control

## 1. Code version control - `Git`

Код-г version control хийх дараалал ба түгээмэл ашиглагддаг коммандууд

Local repository-г эхлүүлэн remote repository-той холбох
- `git init -b main`
- `git add file`
- `git commit -a -m "text"`
- `git remote add origin repo_url`
- `git push origin main`

Remote repository-г хуулан авах
- `git clone remote_url`

Remote repository-с өөрчлөлтийг татах
- `git pull origin main`
- `git fetch origin main`

Branch үүсгэх, branch-тай ажиллах
- `git branch`
- `git branch branch_name`
- `git checkout branch_name`
- `git merge branch_name`

Бусад коммандууд
- `git status`
- `git log`
- `git reset file_name`
- `git tag tag_name`
- `git rm branch_name`

## 2. Data version control - `DVC`

Local data-г version control хийх
- `pip install dvc`
- `dvc init`
- `git commit -m "initial commit"`
- `git branch -M main`
- `git remote add origin remote_url`
- `git push origin main`
- `git tag v1.0`
- `git push --tag`

Data-г remote database руу хуулах
- `pip install dvc`
- `dvc init`
- `git commit -m "initial commit"`
- `git branch -M main`
- `git remote add origin remote_url`
- `dvc remote add remote_name remote_dir`
- `dvc push`
- `git push origin main`
- `git tag v1.0`
- `git push --tag`

Data-г татах
- `git checkout tag`
- `dvc pull`