import { useState } from 'react'
import axios from 'axios'
import './App.css'

function App() {
  const [location, setLocation] = useState('Spain')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleRecommend = async () => {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const response = await axios.post('http://localhost:8000/recommend', {
        location: location,
        user_id: "user_001" // hardcoded for demo
      })
      setResult(response.data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>🍽️ AI Restaurant Recommender</h1>
        <p>Your personal culinary concierge powered by Multi-Agent AI</p>
      </header>

      <main className="main-content">
        <div className="input-section">
          <label htmlFor="location">Where are you dining?</label>
          <input
            id="location"
            type="text"
            value={location}
            onChange={(e) => setLocation(e.target.value)}
            placeholder="e.g. Barcelona, Spain"
          />
          <button onClick={handleRecommend} disabled={loading} className="cta-button">
            {loading ? 'Consulting Experts...' : 'Get Recommendations'}
          </button>
        </div>

        {error && <div className="error-message">Error: {error}</div>}

        {result && (
          <div className="results-section">
            <h2>Your Personalized Selection</h2>
            <div className="recommendation-card">
              <pre className="recommendation-text">{result.recommendations}</pre>
            </div>

            <details className="crew-logs">
              <summary>View Agent Deliberations</summary>
              <pre>{result.crew_log}</pre>
            </details>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
