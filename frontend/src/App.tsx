import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { LandingPage } from './pages/LandingPage';
import { ConnectDevice } from './pages/ConnectDevice';
import { Calibration } from './pages/Calibration';
import { CommunicationHub } from './pages/CommunicationHub';
import { Settings } from './pages/Settings';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/connect" element={<ConnectDevice />} />
        <Route path="/calibrate" element={<Calibration />} />
        <Route path="/communicate" element={<CommunicationHub />} />
        <Route path="/settings" element={<Settings />} />
      </Routes>
    </BrowserRouter>
  );
}
