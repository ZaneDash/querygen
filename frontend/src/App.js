import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [inputText, setInputText] = useState('');
  const [sqlQuery, setSqlQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleInputChange = (e) => {
    setInputText(e.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');
    try {
      const response = await axios.post('http://localhost:5000/predict', { text: inputText });
      setSqlQuery(response.data.sql_query);
    } catch (error) {
      setError('Failed to fetch SQL query. Please try again.');
      console.error('Error fetching SQL query:', error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    const textarea = document.querySelector('.sql-result-textarea');
    if (textarea) {
      textarea.style.height = 'inherit'; 
      textarea.style.height = `${textarea.scrollHeight}px`; 
    }
  }, [sqlQuery]);

  return (
    <div className="App">
      <h1>QueryGen: Seq2Seq SQL Query Generator</h1>
      <div className="reference-info">
        <p> Search for a flight given specific criteria. All criteria are derived from the <strong>ATIS (Airline Travel Information Systems)</strong> dataset, which the generated SQL queries are compatible with. For the reference ATIS schema, visit <a href="https://github.com/jkkummerfeld/text2sql-data/blob/master/data/atis-schema.csv" target="_blank" rel="noopener noreferrer">ATIS Schema</a>.</p>
      </div>
      <form onSubmit={handleSubmit} className="form">
        <textarea
          className="input textarea"
          value={inputText}
          onChange={handleInputChange}
          placeholder="Enter your query"
        />
        <button type="submit" className="button" disabled={isLoading}>
          {isLoading ? 'Generating...' : 'Generate SQL'}
        </button>
      </form>
      {error && <p className="error">{error}</p>}
      {sqlQuery && 
        <textarea 
          className="sql-result-textarea"
          value={sqlQuery}
          readOnly
        />
      }
    </div>
  );
}

export default App;
