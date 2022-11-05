mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"don_yin@outlook.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
[theme]
base="dark"
primaryColor="purple"
" > ~/.streamlit/config.toml