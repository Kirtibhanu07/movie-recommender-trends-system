mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"kirtib.bhanu39@gmail.com\"\n\
"> ~/.streamlit/credentials.toml

echo "\
[server]\n\
port =$PORT\n\
enableCORS = false\n\
headless=true\n\
\n\
"> ~/.streamlit/config.toml
