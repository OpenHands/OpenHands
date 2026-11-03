import React from 'react'
import { Outlet, Link } from 'react-router-dom'

export default function App() {
  return (
    <div className="app">
      <header className="header">
        <h1>Retailer Son</h1>
        <nav>
          <Link to="/">Home</Link>
          <Link to="/scanner">Scanner</Link>
          <Link to="/recent">Recent</Link>
        </nav>
      </header>
      <main>
        <Outlet />
      </main>
      <footer className="footer">Retailer Son — Test MVP</footer>
    </div>
  )
}
