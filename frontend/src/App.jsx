import { useState } from "react";
import axios from "axios";
import "./App.css";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

function App() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState("");

  const uploadLogs = async () => {
    try {
      setUploadStatus("Uploading...");
      await axios.post(`${API_URL}/upload-logs`);
      setUploadStatus("Logs uploaded successfully!");
    } catch (error) {
      setUploadStatus("Upload failed. Is the backend running?");
    }
  };

  const analyzeIncident = async () => {
    if (!question.trim()) return;
    try {
      setLoading(true);
      const response = await axios.post(
        `${API_URL}/analyze-incident`,
        { question }
      );
      setResult(response.data);
    } catch (error) {
      setResult({ error: "Analysis failed. Make sure logs are uploaded first." });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1>🔍 TraceOps AI RCA System</h1>
        <p>AI-powered log analysis and root cause identification</p>
      </header>

      <main className="main">
        <section className="upload-section">
          <button onClick={uploadLogs} className="btn btn-upload">
            📤 Upload Sample Logs
          </button>
          {uploadStatus && <span className="status">{uploadStatus}</span>}
        </section>

        <section className="query-section">
          <input
            type="text"
            placeholder="Ask: Why is inference latency high?"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && analyzeIncident()}
            className="input-query"
          />
          <button
            onClick={analyzeIncident}
            disabled={loading}
            className="btn btn-analyze"
          >
            {loading ? "Analyzing..." : "🧠 Analyze Incident"}
          </button>
        </section>

        {result && !result.error && (
          <section className="results">
            <div className="result-card">
              <h3>📋 Related Logs</h3>
              <ul>
                {result.related_logs.map((log, index) => (
                  <li key={index} className="log-item">{log}</li>
                ))}
              </ul>
            </div>

            <div className="result-card root-cause">
              <h3>🔴 Root Cause</h3>
              <p>{result.root_cause}</p>
            </div>

            <div className="result-card suggested-fix">
              <h3>✅ Suggested Fix</h3>
              <p>{result.suggested_fix}</p>
            </div>
          </section>
        )}

        {result && result.error && (
          <section className="results">
            <div className="result-card error">
              <p>{result.error}</p>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
