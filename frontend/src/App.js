import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const API_URL = 'http://localhost:8000';

function App() {
  const [documents, setDocuments] = useState([]);
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [kValue, setKValue] = useState(3);
  const [message, setMessage] = useState('');
  const [sessionId] = useState(() => 'session_' + Date.now()); // Generate unique session ID
  const [chatHistory, setChatHistory] = useState([]); // Store chat messages

  // Fetch documents on load
  useEffect(() => {
    fetchDocuments();
  }, []);

  const fetchDocuments = async () => {
    try {
      const response = await axios.get(`${API_URL}/documents`);
      setDocuments(response.data);
    } catch (error) {
      console.error('Error fetching documents:', error);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    setUploading(true);
    setMessage('');

    try {
      const response = await axios.post(`${API_URL}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      setMessage(response.data.message);
      fetchDocuments();
      event.target.value = ''; // Reset file input
    } catch (error) {
      setMessage(`Error: ${error.response?.data?.detail || error.message}`);
    } finally {
      setUploading(false);
    }
  };

  const handleQuery = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;

    setLoading(true);
    
    // Add user question to chat history
    const userMessage = { type: 'question', text: question };
    setChatHistory(prev => [...prev, userMessage]);

    try {
      const response = await axios.post(`${API_URL}/query`, {
        question: question,
        k: kValue,
        session_id: sessionId,
      });
      
      const botAnswer = response.data.answer;
      setAnswer(botAnswer);
      
      // Add bot answer to chat history
      const botMessage = { type: 'answer', text: botAnswer };
      setChatHistory(prev => [...prev, botMessage]);
      
      // Clear question input
      setQuestion('');
    } catch (error) {
      const errorMsg = `Error: ${error.response?.data?.detail || error.message}`;
      setAnswer(errorMsg);
      setChatHistory(prev => [...prev, { type: 'error', text: errorMsg }]);
    } finally {
      setLoading(false);
    }
  };

  const handleClearAll = async () => {
    if (!window.confirm('Are you sure you want to clear all documents?')) return;

    try {
      await axios.delete(`${API_URL}/documents`);
      setDocuments([]);
      setAnswer('');
      setChatHistory([]);
      setMessage('All documents cleared successfully');
    } catch (error) {
      setMessage(`Error: ${error.response?.data?.detail || error.message}`);
    }
  };

  return (
    <div className="App">
      <div className="container">
        <header className="header">
          <h1>🤖 RAG Document Q&A</h1>
          <p>Multi-Document AI Assistant with Persistence</p>
        </header>

        <div className="main-content">
          {/* Sidebar */}
          <aside className="sidebar">
            <h2>📚 Documents</h2>
            
            <div className="upload-section">
              <label htmlFor="file-upload" className="upload-btn">
                {uploading ? 'Uploading...' : '+ Upload PDF'}
              </label>
              <input
                id="file-upload"
                type="file"
                accept=".pdf"
                onChange={handleFileUpload}
                disabled={uploading}
                style={{ display: 'none' }}
              />
            </div>

            {message && (
              <div className={`message ${message.includes('Error') ? 'error' : 'success'}`}>
                {message}
              </div>
            )}

            <div className="documents-list">
              {documents.length === 0 ? (
                <p className="no-docs">No documents yet</p>
              ) : (
                documents.map((doc, index) => (
                  <div key={index} className="document-item">
                    ✓ {doc}
                  </div>
                ))
              )}
            </div>

            {documents.length > 0 && (
              <button className="clear-btn" onClick={handleClearAll}>
                🗑️ Clear All
              </button>
            )}
          </aside>

          {/* Main Query Area */}
          <main className="query-section">
            <h2>💬 Ask Questions</h2>
            
            {documents.length === 0 ? (
              <div className="empty-state">
                <p>👆 Upload a PDF document to get started!</p>
              </div>
            ) : (
              <>
                <form onSubmit={handleQuery} className="query-form">
                  <div className="input-group">
                    <input
                      type="text"
                      value={question}
                      onChange={(e) => setQuestion(e.target.value)}
                      placeholder="Ask a question about your documents..."
                      className="query-input"
                      disabled={loading}
                    />
                    
                    <div className="k-value-selector">
                      <label htmlFor="k-value">Top-K:</label>
                      <input
                        id="k-value"
                        type="number"
                        min="1"
                        max="10"
                        value={kValue}
                        onChange={(e) => setKValue(parseInt(e.target.value))}
                        className="k-input"
                      />
                    </div>
                  </div>
                  
                  <button 
                    type="submit" 
                    className="submit-btn"
                    disabled={loading || !question.trim()}
                  >
                    {loading ? 'Searching...' : 'Ask'}
                  </button>
                </form>

                {chatHistory.length > 0 && (
                  <div className="chat-history">
                    <h3>💬 Conversation:</h3>
                    <div className="chat-messages">
                      {chatHistory.map((msg, index) => (
                        <div key={index} className={`message ${msg.type}`}>
                          {msg.type === 'question' && <strong>You:</strong>}
                          {msg.type === 'answer' && <strong>Assistant:</strong>}
                          {msg.type === 'error' && <strong>Error:</strong>}
                          <p>{msg.text}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {loading && (
                  <div className="loading-indicator">
                    <p>🔍 Thinking...</p>
                  </div>
                )}
              </>
            )}
          </main>
        </div>
      </div>
    </div>
  );
}

export default App;
