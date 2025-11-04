import React, { useState, useEffect } from 'react';
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

function AnalysisComponent() {
  const [symbol, setSymbol] = useState('AAPL');
  const [workflowType, setWorkflowType] = useState('agentic');
  const [focus, setFocus] = useState('comprehensive');
  const [workflows, setWorkflows] = useState([]);
  const [focusTypes, setFocusTypes] = useState([]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Fetch available workflows and focus types on mount
  useEffect(() => {
    fetchWorkflows();
    fetchFocusTypes();
  }, []);

  const fetchWorkflows = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/workflows`);
      setWorkflows(response.data.workflows);
    } catch (err) {
      console.error('Error fetching workflows:', err);
    }
  };

  const fetchFocusTypes = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/focus-types`);
      setFocusTypes(response.data.focus_types);
    } catch (err) {
      console.error('Error fetching focus types:', err);
    }
  };

  const handleAnalyze = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await axios.post(`${API_BASE_URL}/api/analyze`, {
        symbol: symbol.toUpperCase(),
        workflow_type: workflowType,
        focus: focus
      });

      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Analysis failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '20px', maxWidth: '1200px', margin: '0 auto' }}>
      <h1>📊 Multi-Agent Financial Analysis</h1>
      
      {/* Input Form */}
      <div style={{ 
        background: '#f5f5f5', 
        padding: '20px', 
        borderRadius: '8px',
        marginBottom: '20px'
      }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr auto', gap: '15px', alignItems: 'end' }}>
          <div>
            <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
              Stock Symbol
            </label>
            <input
              type="text"
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
              placeholder="AAPL"
              style={{ width: '100%', padding: '8px', fontSize: '14px' }}
            />
          </div>

          <div>
            <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
              Workflow Type
            </label>
            <select
              value={workflowType}
              onChange={(e) => setWorkflowType(e.target.value)}
              style={{ width: '100%', padding: '8px', fontSize: '14px' }}
            >
              {workflows.map(wf => (
                <option key={wf.id} value={wf.id}>{wf.name}</option>
              ))}
            </select>
          </div>

          <div>
            <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
              Analysis Focus
            </label>
            <select
              value={focus}
              onChange={(e) => setFocus(e.target.value)}
              style={{ width: '100%', padding: '8px', fontSize: '14px' }}
            >
              {focusTypes.map(ft => (
                <option key={ft.id} value={ft.id}>{ft.name}</option>
              ))}
            </select>
          </div>

          <button
            onClick={handleAnalyze}
            disabled={loading || !symbol}
            style={{
              padding: '10px 20px',
              fontSize: '16px',
              backgroundColor: '#007bff',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: loading ? 'not-allowed' : 'pointer',
              opacity: loading ? 0.6 : 1
            }}
          >
            {loading ? 'Analyzing...' : '🚀 Analyze'}
          </button>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div style={{
          background: '#fee',
          color: '#c33',
          padding: '15px',
          borderRadius: '4px',
          marginBottom: '20px'
        }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* Results Display */}
      {result && (
        <div style={{ marginTop: '20px' }}>
          <h2>Analysis Results</h2>
          
          {/* Status Indicator */}
          <div style={{
            padding: '15px',
            borderRadius: '4px',
            marginBottom: '20px',
            backgroundColor: result.status === 'success' ? '#d4edda' : '#f8d7da',
            color: result.status === 'success' ? '#155724' : '#721c24'
          }}>
            <strong>Status:</strong> {result.status.toUpperCase()}
            <br />
            <strong>Symbol:</strong> {result.symbol}
            <br />
            <strong>Workflow:</strong> {result.workflow_type}
            <br />
            <strong>Timestamp:</strong> {result.timestamp}
          </div>

          {/* Summary */}
          {result.result?.summary && (
            <div style={{
              background: '#eef6ff',
              padding: '15px',
              borderRadius: '4px',
              marginBottom: '20px',
              border: '1px solid #cfe2ff'
            }}>
              <h3>Summary</h3>
              <pre style={{
                whiteSpace: 'pre-wrap',
                background: '#f7fbff',
                padding: '10px',
                borderRadius: '4px',
                overflow: 'auto'
              }}>
                {result.result.summary}
              </pre>
            </div>
          )}

          {/* Execution Path */}
          {result.result?.nodes_executed && (
            <div style={{
              background: '#f8f9fa',
              padding: '15px',
              borderRadius: '4px',
              marginBottom: '20px'
            }}>
              <h3>Execution Path</h3>
              <p>
                {result.result.nodes_executed.map((node, idx) => (
                  <span key={idx}>
                    <strong>{node.replace('_', ' ')}</strong>
                    {idx < result.result.nodes_executed.length - 1 && ' → '}
                  </span>
                ))}
              </p>
              <p><strong>Nodes Executed:</strong> {result.result.nodes_executed.length}</p>
            </div>
          )}

          {/* Results by Workflow Type */}
          {result.result?.results && (
            <div>
              <h3>Detailed Results</h3>
              
              {result.workflow_type === 'investment_agent' && result.result.results.investment_agent && (
                <div style={{
                  background: 'white',
                  padding: '15px',
                  borderRadius: '4px',
                  marginBottom: '15px',
                  border: '1px solid #ddd'
                }}>
                  <h4>Investment Agent Analysis</h4>
                  <div style={{ marginBottom: '10px' }}>
                    <strong>Plan</strong>
                    <pre style={{
                      background: '#f5f5f5',
                      padding: '10px',
                      borderRadius: '4px',
                      whiteSpace: 'pre-wrap'
                    }}>{result.result.results.investment_agent.plan || 'N/A'}</pre>
                  </div>
                  <div style={{ marginBottom: '10px' }}>
                    <strong>Analysis</strong>
                    <pre style={{
                      background: '#f5f5f5',
                      padding: '10px',
                      borderRadius: '4px',
                      whiteSpace: 'pre-wrap'
                    }}>{result.result.results.investment_agent.analysis || 'N/A'}</pre>
                  </div>
                  <div>
                    <strong>Reflection</strong>
                    <pre style={{
                      background: '#f5f5f5',
                      padding: '10px',
                      borderRadius: '4px',
                      whiteSpace: 'pre-wrap'
                    }}>{result.result.results.investment_agent.reflection?.reflection || 'N/A'}</pre>
                  </div>
                </div>
              )}

              {result.workflow_type === 'routing' && result.result.results.routing && (
                <div style={{
                  background: 'white',
                  padding: '15px',
                  borderRadius: '4px',
                  marginBottom: '15px',
                  border: '1px solid #ddd'
                }}>
                  <h4>Routing Workflow Results</h4>
                  <p><strong>Specialists Used:</strong> {result.result.results.routing.specialists_used?.join(', ')}</p>
                  <pre style={{
                    background: '#f5f5f5',
                    padding: '10px',
                    borderRadius: '4px',
                    overflow: 'auto',
                    maxHeight: '400px'
                  }}>
                    {JSON.stringify(result.result.results.routing, null, 2)}
                  </pre>
                </div>
              )}

              {/* Full JSON View */}
              <details style={{ marginTop: '20px' }}>
                <summary style={{ cursor: 'pointer', fontWeight: 'bold' }}>
                  View Full JSON Response
                </summary>
                <pre style={{
                  background: '#f5f5f5',
                  padding: '15px',
                  borderRadius: '4px',
                  overflow: 'auto',
                  maxHeight: '600px',
                  marginTop: '10px'
                }}>
                  {JSON.stringify(result, null, 2)}
                </pre>
              </details>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default AnalysisComponent;

