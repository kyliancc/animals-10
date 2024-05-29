if [ ! -d "./animals10/" ]; then
  kaggle datasets download -d alessiocorrado99/animals10
  unzip -d ./animals10/ ./animals10.zip
  rm animals10.zip
fi
python split.py
