This project leverages a fine-tuned pretrained BART transformer to generate SQL queries from flight searches. Flights can be specified based on the criteria defined in the ATIS dataset, with schema [linked here](https://github.com/jkkummerfeld/text2sql-data/blob/master/data/atis-schema.csv). React and Flask are used as the frontend and backend.

To open the project, install all required packages with `pip install -r requirements.txt` and start the backend with `flask run`. Next, navigate to the frontend folder and run `npm start`. After the frontend opens in a browser window, type flight searches into the text box.

Work used to develop the model deployed in this project, as well as preliminary manually-implemented models, is in the modelDevelopment.ipynb file in the folder modelDevelopment.




