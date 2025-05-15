# 💬 Sales Forecasting & Q&A Assistant

A powerful and interactive Streamlit app that lets business users and executives query forecasted sales data using natural language — no SQL required.

---

## 🔍 Why This Project?

Data is only useful if it's accessible.

In many organizations, important data is stored in SQL databases but only technical users can extract insights from them. This tool was created to **empower non-technical decision-makers** — especially executives — to directly query a sales forecasting database using plain English.

Whether you're asking:
- “What were the predicted sales for Store 1 in August 2017?”
- “Show average forecasted sales by family”

…the app translates these questions into SQL, fetches the result, and shows the output along with the underlying query.

---

## 🚀 Live App

👉 Try the app here:  
[**🔗 Launch on Streamlit Cloud**](https://dhruvvaghani-sales-forecasting-app.streamlit.app)

---

## 🧠 Features

- 🔗 Natural language to SQL using LangChain + OpenAI
- 📊 Interprets questions and shows forecasted sales predictions
- 🧠 Fallback XGBoost model for unseen queries
- 🖼️ Clean, interactive Streamlit UI
- 🔒 Secure API key handling with Streamlit Secrets

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [OpenAI API](https://platform.openai.com/)
- [SQLite](https://www.sqlite.org/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [Pandas + Python Ecosystem](https://pandas.pydata.org/)

---

## 💡 How to Use the App

1. Open the [Live App](https://dhruvvaghani-sales-forecasting-app.streamlit.app)
2. Type a question in the input box (e.g., _“Total forecasted sales in August 2017”_)
3. View:
   - ✅ The plain-text result
   - 🧠 The SQL query generated
   - 📋 Any data tables returned
4. If the query doesn’t match existing database rows, a fallback model gives a **predicted answer** instead.

---

## 🤝 Contributing

Feel free to **clone this repo**, explore the code, and contribute!  
Pull requests, improvements, and ideas are always welcome.

```bash
git clone https://github.com/DhruvVaghani/sales-forecasting-app.git

📬 Contact
Have questions or suggestions? Open an issue or reach out on LinkedIn.

⭐ If you found this useful, don’t forget to star the repo!
