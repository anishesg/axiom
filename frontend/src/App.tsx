import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { LandingPage } from './pages/LandingPage';
import { ConnectDevice } from './pages/ConnectDevice';
import { Calibration } from './pages/Calibration';
import { ConversationMode } from './pages/ConversationMode';
import { Settings } from './pages/Settings';
import { LiveDecode } from './pages/LiveDecode';
import { Dashboard } from './pages/Dashboard';
import { TrainingMode } from './pages/TrainingMode';
import { Team } from './pages/Team';
import { Research } from './pages/Research';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/connect" element={<ConnectDevice />} />
        <Route path="/calibrate" element={<Calibration />} />
                <Route path="/conversation" element={<ConversationMode />} />
        <Route path="/decode" element={<LiveDecode />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/train" element={<TrainingMode />} />
        <Route path="/settings" element={<Settings />} />
        <Route path="/team" element={<Team />} />
        <Route path="/research" element={<Research />} />
      </Routes>
    </BrowserRouter>
  );
}
