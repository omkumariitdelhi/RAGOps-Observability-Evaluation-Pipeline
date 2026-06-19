import React, { useState } from 'react';
import axios from 'axios';
import { Search, Cpu, Database, Send, Network, MessageSquareDashed } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

function App() {
  const [query, setQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [response, setResponse] = useState(null);
  const [selectedModel, setSelectedModel] = useState('');

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setIsLoading(true);
    setResponse(null);

    try {
      const res = await axios.post('http://127.0.0.1:8000/query', { query });
      setResponse(res.data);
      if (res.data.results) {
        setSelectedModel(Object.keys(res.data.results)[0]);
      }
    } catch (error) {
      console.error("Error fetching data:", error);
      alert("Failed to query the backend. Ensure FastAPI is running on port 8000.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-wrapper">
      <header className="top-navbar">
        <div className="brand-title">
          <Network className="brand-icon" size={28} />
          Enterprise Knowledge AI
        </div>
      </header>

      <main className="main-content">
        {/* Left Column: Chat and Models */}
        <div className="chat-column">
          <div className="search-wrapper">
            <form onSubmit={handleSearch} className="search-box">
              <Search className="text-secondary" style={{ color: '#94a3b8' }} />
              <input
                id="search-input"
                type="text"
                className="search-input"
                placeholder="Ask a technical question about the knowledge base..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                disabled={isLoading}
              />
              <button id="query-button" type="submit" className="btn-primary" disabled={isLoading || !query.trim()}>
                <Send size={18} />
                Ask AI
              </button>
            </form>
          </div>

          <div className="models-feed">
            {isLoading && (
              <div className="skeleton-card">
                <div className="skeleton-line title"></div>
                <div className="skeleton-line w-full"></div>
                <div className="skeleton-line w-5-6"></div>
                <div className="skeleton-line w-4-6"></div>
                <div className="skeleton-line w-full" style={{ marginTop: '1rem' }}></div>
                <div className="skeleton-line w-5-6"></div>
              </div>
            )}

            {!isLoading && !response && (
              <div className="empty-state">
                <MessageSquareDashed size={48} className="empty-icon" />
                <h2>Ready to assist</h2>
                <p style={{ marginTop: '0.5rem' }}>Enter a query above to search the document database across multiple LLMs simultaneously.</p>
              </div>
            )}

            {!isLoading && response && response.results && (
              <div className="model-view-container">
                <div className="model-selector-wrapper">
                  <label htmlFor="model-select" className="model-selector-label">Select AI Model:</label>
                  <select 
                    id="model-select" 
                    className="model-dropdown"
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                  >
                    {Object.keys(response.results).map((modelName) => (
                      <option key={modelName} value={modelName}>{modelName}</option>
                    ))}
                  </select>
                </div>
                
                {selectedModel && response.results[selectedModel] && (
                  <div className="model-card active">
                    <div className="model-card-header">
                      <Cpu className="model-icon" size={20} />
                      <span className="model-badge">{selectedModel}</span>
                    </div>
                    <div className="markdown-body">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {response.results[selectedModel]}
                      </ReactMarkdown>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Context Sidebar */}
        <aside className="context-sidebar">
          <div className="context-header">
            <Database size={20} className="context-icon" />
            Retrieved Context
          </div>
          <div className="context-list">
            {isLoading ? (
              <>
                <div className="skeleton-card" style={{ padding: '1rem', border: 'none', background: 'rgba(255,255,255,0.02)' }}>
                  <div className="skeleton-line w-full"></div>
                  <div className="skeleton-line w-5-6"></div>
                </div>
                <div className="skeleton-card" style={{ padding: '1rem', border: 'none', background: 'rgba(255,255,255,0.02)', animationDelay: '0.1s' }}>
                  <div className="skeleton-line w-full"></div>
                  <div className="skeleton-line w-4-6"></div>
                </div>
              </>
            ) : response?.contexts && response.contexts.length > 0 ? (
              response.contexts.map((ctx, idx) => (
                <div className="context-chunk" key={idx}>
                  <span className="chunk-label">Chunk {idx + 1}</span>
                  {ctx}
                </div>
              ))
            ) : (
              <div className="empty-state" style={{ padding: '2rem 1rem' }}>
                <Database size={32} className="empty-icon" />
                <p>No context retrieved yet.</p>
              </div>
            )}
          </div>
        </aside>
      </main>
    </div>
  );
}

export default App;
