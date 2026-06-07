import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Layout } from '../components/layout/Layout';

interface ToggleProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
}

function Toggle({ checked, onChange }: ToggleProps) {
  return (
    <label className="relative inline-flex items-center cursor-pointer">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="sr-only peer"
      />
      <div className="w-14 h-7 bg-surface-variant rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-0.5 after:left-[4px] after:bg-white after:rounded-full after:h-6 after:w-6 after:transition-all peer-checked:bg-primary" />
    </label>
  );
}

interface StepperProps {
  value: string;
  onIncrement: () => void;
  onDecrement: () => void;
}

function Stepper({ value, onIncrement, onDecrement }: StepperProps) {
  return (
    <div className="flex items-center border border-surface-variant">
      <button
        onClick={onDecrement}
        className="w-16 h-16 flex items-center justify-center hover:bg-surface-container-low transition-colors active:bg-surface-container"
      >
        <span className="material-symbols-outlined">remove</span>
      </button>
      <div className="w-20 text-center text-headline-md border-x border-surface-variant">
        {value}
      </div>
      <button
        onClick={onIncrement}
        className="w-16 h-16 flex items-center justify-center hover:bg-surface-container-low transition-colors active:bg-surface-container"
      >
        <span className="material-symbols-outlined">add</span>
      </button>
    </div>
  );
}

export function Settings() {
  const navigate = useNavigate();

  // Settings state
  const [autoCalibration, setAutoCalibration] = useState(true);
  const [textSize, setTextSize] = useState(24);
  const [dwellTime, setDwellTime] = useState(1.2);
  const [audioFeedback, setAudioFeedback] = useState(false);
  const [usageAnalytics, setUsageAnalytics] = useState(true);

  const handleRecalibrate = () => {
    navigate('/calibrate');
  };

  const handleDisconnect = () => {
    navigate('/connect');
  };

  return (
    <Layout showBackButton backTo="/communicate" backLabel="Back to Chat" showNav={false}>
      <main className="flex-grow w-full max-w-[1200px] mx-auto px-margin-page py-stack-lg">
        <div className="max-w-[800px] mx-auto">
          <h1 className="text-headline-lg mb-stack-lg tracking-tight">System Settings</h1>

          {/* Settings Sections */}
          <div className="flex flex-col gap-stack-lg">
            {/* Vocabulary Section */}
            <section>
              <h2 className="text-label-lg text-on-surface-variant uppercase tracking-widest mb-stack-sm">
                Vocabulary
              </h2>
              <div className="border-t border-surface-variant">
                <button className="w-full flex justify-between items-center py-6 hover:bg-surface-container-low transition-colors active:scale-[0.99]">
                  <span className="text-headline-md">Personal Dictionary</span>
                  <span className="material-symbols-outlined text-on-surface-variant">
                    chevron_right
                  </span>
                </button>
                <div className="border-t border-surface-variant" />
                <button className="w-full flex justify-between items-center py-6 hover:bg-surface-container-low transition-colors active:scale-[0.99]">
                  <span className="text-headline-md">Quick Phrases</span>
                  <span className="material-symbols-outlined text-on-surface-variant">
                    chevron_right
                  </span>
                </button>
              </div>
            </section>

            {/* Calibration Section */}
            <section>
              <h2 className="text-label-lg text-on-surface-variant uppercase tracking-widest mb-stack-sm">
                Calibration
              </h2>
              <div className="border-t border-surface-variant">
                <div className="flex flex-col md:flex-row md:items-center justify-between py-6 gap-4">
                  <div>
                    <h3 className="text-headline-md">Signal Strength</h3>
                    <p className="text-on-surface-variant text-body-md">Optimal (98%)</p>
                  </div>
                  <button
                    onClick={handleRecalibrate}
                    className="px-8 py-3 border border-primary text-label-lg hover:bg-primary hover:text-on-primary transition-all active:scale-[0.98]"
                  >
                    Recalibrate Now
                  </button>
                </div>
                <div className="border-t border-surface-variant" />
                <div className="flex justify-between items-center py-6">
                  <span className="text-headline-md">Auto-Calibration</span>
                  <Toggle checked={autoCalibration} onChange={setAutoCalibration} />
                </div>
              </div>
            </section>

            {/* Accessibility Section */}
            <section>
              <h2 className="text-label-lg text-on-surface-variant uppercase tracking-widest mb-stack-sm">
                Accessibility
              </h2>
              <div className="border-t border-surface-variant">
                {/* Text Size */}
                <div className="flex flex-col md:flex-row md:items-center justify-between py-6 gap-4">
                  <span className="text-headline-md">Text Size</span>
                  <Stepper
                    value={`${textSize}px`}
                    onIncrement={() => setTextSize((s) => Math.min(s + 2, 48))}
                    onDecrement={() => setTextSize((s) => Math.max(s - 2, 12))}
                  />
                </div>
                <div className="border-t border-surface-variant" />

                {/* Dwell Time */}
                <div className="flex flex-col md:flex-row md:items-center justify-between py-6 gap-4">
                  <div>
                    <h3 className="text-headline-md">Dwell Time</h3>
                    <p className="text-on-surface-variant text-body-md">
                      Duration for selection confirmation
                    </p>
                  </div>
                  <Stepper
                    value={`${dwellTime.toFixed(1)}s`}
                    onIncrement={() => setDwellTime((t) => Math.min(t + 0.1, 5.0))}
                    onDecrement={() => setDwellTime((t) => Math.max(t - 0.1, 0.5))}
                  />
                </div>
                <div className="border-t border-surface-variant" />

                {/* Audio Feedback */}
                <div className="flex justify-between items-center py-6">
                  <span className="text-headline-md">Audio Feedback</span>
                  <Toggle checked={audioFeedback} onChange={setAudioFeedback} />
                </div>
              </div>
            </section>

            {/* Device Section */}
            <section>
              <h2 className="text-label-lg text-on-surface-variant uppercase tracking-widest mb-stack-sm">
                Device
              </h2>
              <div className="border-t border-surface-variant">
                <div className="flex justify-between items-center py-6">
                  <div>
                    <h3 className="text-headline-md">Muse S Headband</h3>
                    <p className="text-on-surface-variant text-body-md">
                      Firmware: v2.4.0 • Battery: 84%
                    </p>
                  </div>
                  <span className="material-symbols-outlined text-on-surface-variant">
                    bluetooth_connected
                  </span>
                </div>
                <div className="border-t border-surface-variant" />
                <button
                  onClick={handleDisconnect}
                  className="w-full text-left py-6 text-error text-headline-md hover:bg-surface-container-low transition-colors active:scale-[0.99]"
                >
                  Disconnect Device
                </button>
              </div>
            </section>

            {/* Privacy Section */}
            <section>
              <h2 className="text-label-lg text-on-surface-variant uppercase tracking-widest mb-stack-sm">
                Privacy
              </h2>
              <div className="border-t border-surface-variant">
                <button className="w-full flex justify-between items-center py-6 hover:bg-surface-container-low transition-colors active:scale-[0.99]">
                  <span className="text-headline-md">Data Encryption Settings</span>
                  <span className="material-symbols-outlined text-on-surface-variant">
                    chevron_right
                  </span>
                </button>
                <div className="border-t border-surface-variant" />
                <div className="flex justify-between items-center py-6">
                  <span className="text-headline-md">Usage Analytics</span>
                  <Toggle checked={usageAnalytics} onChange={setUsageAnalytics} />
                </div>
              </div>
            </section>
          </div>
        </div>
      </main>
    </Layout>
  );
}
