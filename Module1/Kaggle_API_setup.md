# Дараах алхмуудыг ашиглан Kaggle API-г terminal-с ашиглана

1. Kaggle аккаунт руугаа ороод kaggle token бүхий `kaggle.json` файлыг татаж авах.
1. Kaggle санг суулгахын тулд Terminal дээр `pip install kaggle` комманд-г уншуулах
1. Шинэ нуусан kaggle гэдэг нэртэй хавтас нээх `mkdir $HOME/.kaggle`
1. kaggle.json файлыг хавтас руу нүүлгэх mv kaggle.json `$HOME/.kaggle/kaggle.json`
1. KAGGLE_CONFIG_DIR замыг нэмж өгөх export `KAGGLE_CONFIG_DIR=$HOME/.kaggle/`
1. kaggle.json файлыг оруулсан замыг нэмэх xport `KAGGLE_CONFIG_DIR=$KAGGLE_CONFIG_DIR/kaggle.json`
1. kaggle.json-д унших эрх өгөх `chmod 600 $KAGGLE_CONFIG_DIR`
1. Kaggle username-г нэмэх `export KAGGLE_USERNAME=username`-г [энд](Setting_Env_var_setup.md) заасан файлд бичиж, уншуулна.
1. Kaggle token-г нэмэх `export KAGGLE_KEY=xxxxxxxx"`-г [энд](Setting_Env_var_setup.md) заасан файлд бичиж, уншуулна.
1. Өгөгдөл хадгалах `$HOME/MLData` хавтас нээх
1. Өгөгдөл хадгалах directory-г environment руу өгөх `export DATA_DIR=$HOME/MLData` -г [энд](Setting_Env_var_setup.md) заасан файлд бичиж, уншуулна.
1. Kaggle dataset-н вебсайт руу орон дата-н `Copy API Command`-г хуулан `-p $DATADIR`-г ард нь нэмж уншуулна. `kaggle datasets download -d joebeachcapital/30000-spotify-songs -p $DATADIR` 
1. Хэрэв өгөгдөл zip байвал unzip хийх `unzip $DATA_DIR/30000-spotify-songs.zip -d $DATA_DIR/Spotify/`
1. Zip файл байгаа эсэхийг шалгах `find $DATA_DIR -name "*spotify*.zip"`
1. Zip файлыг устгах `rm $DATA_DIR/*spotify*.zip`


Эх үүсвэр: https://github.com/Kaggle/kaggle-api