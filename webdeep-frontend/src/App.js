import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import HomePage from './components/HomePage';
import ImagePredictor from './components/ImagePredictor';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/predict" element={<ImagePredictor />} />
      </Routes>
    </Router>
  );
}

export default App;