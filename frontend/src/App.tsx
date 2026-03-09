import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, GlobalStyles } from '@mui/material';
import CssBaseline from '@mui/material/CssBaseline';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Predict from './pages/Predict';
import { theme } from './theme';

const globalStyles = (
  <GlobalStyles
    styles={{
      body: {
        backgroundColor: '#070A11',
        backgroundImage: 
          'radial-gradient(circle at 15% 50%, rgba(0, 230, 118, 0.04), transparent 25%), radial-gradient(circle at 85% 30%, rgba(41, 121, 255, 0.04), transparent 25%)',
        backgroundAttachment: 'fixed',
        margin: 0,
        padding: 0,
        color: '#F3F4F6',
      },
      '*::-webkit-scrollbar': {
        width: '8px',
      },
      '*::-webkit-scrollbar-track': {
        background: '#070A11',
      },
      '*::-webkit-scrollbar-thumb': {
        background: '#1F2937',
        borderRadius: '4px',
      },
      '*::-webkit-scrollbar-thumb:hover': {
        background: '#374151',
      },
    }}
  />
);

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      {globalStyles}
      <Router>
        <Navbar />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/predict" element={<Predict />} />
        </Routes>
      </Router>
    </ThemeProvider>
  );
}

export default App;