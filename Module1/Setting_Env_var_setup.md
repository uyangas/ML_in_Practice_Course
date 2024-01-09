# Environment Variable-г permanently хадгалах

1. Mac, Linux дээр ажиллаж байгаа бол

    1. `zsh` ашиглаж байгаа бол
        - файл байгаа эсэхийг шалгах `find ".zprofile"`
        - файл-н access хийх эрхийг шалгах `ls -l ~/.zprofile`
        - файл-г edit хийх `nano ~/.zprofile`
        - файл-г уншуулан идэвхижүүлэх `source ~/.zprofile`

    2. `bash` ашиглаж байгаа бол
        - файл байгаа эсэхийг шалгах `find ".bash_profile"`
        - файл-н access хийх эрхийг шалгах `ls -l ~/.bash_profile`
        - файл-г edit хийх `nano ~/.bash_profile`
        - файл-г уншуулан идэвхижүүлэх `source ~/.bash_profile`

2. Windows дээр ажиллаж байгаа бол
    - Зөвхөн тухайн session-д ашиглах бол `set MY_VARIABLE "value"`
    - Бүх session-д ашиглах бол `setx MY_VARIABLE "value"`