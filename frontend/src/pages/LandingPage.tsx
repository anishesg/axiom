import { Link } from 'react-router-dom';
import { Layout } from '../components/layout/Layout';
import { SignalIndicator } from '../components/ui/SignalIndicator';

export function LandingPage() {
  return (
    <Layout>
      {/* Hero Section */}
      <section className="min-h-[80vh] flex flex-col items-center justify-center bg-surface-container-lowest px-margin-page py-stack-lg">
        <div className="max-w-[900px] mx-auto text-center">
          {/* Logo */}
          <div className="mb-stack-md">
            <span className="text-display text-primary font-bold tracking-tighter">ELEVEN</span>
          </div>

          {/* Headline */}
          <h1 className="text-display md:text-[64px] mb-6 tracking-tighter font-bold leading-[1.1]">
            Communication through thought.
          </h1>

          {/* Subtitle */}
          <p className="text-body-lg md:text-[22px] text-on-surface-variant max-w-2xl mx-auto mb-stack-md leading-relaxed">
            Eleven is a brain-computer interface that enables people with severe motor disabilities
            to communicate using only their thoughts. No voice. No hands. Just intention.
          </p>

          {/* CTAs */}
          <div className="flex flex-col md:flex-row items-center justify-center gap-stack-sm">
            <Link
              to="/connect"
              className="w-full md:w-auto min-w-[200px] min-h-target-min bg-primary text-on-primary text-label-lg px-8 py-4 transition-all hover:opacity-90 active:scale-95 border border-primary flex items-center justify-center font-semibold"
            >
              Get Started
            </Link>
            <a
              href="#how-it-works"
              className="text-label-lg text-primary flex items-center gap-2 px-8 py-4 hover:underline transition-all font-semibold"
            >
              Learn how it works
              <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>
                arrow_downward
              </span>
            </a>
          </div>

          {/* Signal Status */}
          <div className="mt-stack-lg">
            <SignalIndicator status="ready" label="System Ready" />
          </div>
        </div>
      </section>

      {/* Mission Section */}
      <section className="py-stack-lg px-margin-page bg-white border-t border-secondary-container">
        <div className="max-w-[900px] mx-auto">
          <p className="text-label-lg uppercase tracking-widest text-on-surface-variant mb-4">Our Mission</p>
          <h2 className="text-headline-lg md:text-[40px] font-bold tracking-tight mb-6">
            Restoring voice to those who have lost it.
          </h2>
          <p className="text-body-lg text-on-surface-variant leading-relaxed mb-6">
            Every year, thousands of people lose the ability to speak due to ALS, stroke, traumatic brain injury,
            or other conditions. Current assistive technologies require precise motor control that many patients
            simply don't have.
          </p>
          <p className="text-body-lg text-on-surface-variant leading-relaxed">
            Eleven changes this. Using a consumer-grade EEG headband (Muse S), we detect neural signals
            associated with deliberate mental actions—jaw clenches, focused attention, relaxation states—and
            translate them into communication. No eye tracking required. No finger movement needed.
          </p>
        </div>
      </section>

      {/* How It Works Section */}
      <section id="how-it-works" className="py-stack-lg px-margin-page bg-surface-container-lowest border-t border-secondary-container">
        <div className="max-w-[1100px] mx-auto">
          <p className="text-label-lg uppercase tracking-widest text-on-surface-variant mb-4">How It Works</p>
          <h2 className="text-headline-lg md:text-[40px] font-bold tracking-tight mb-stack-md">
            Three simple steps to communicate.
          </h2>

          <div className="grid md:grid-cols-3 gap-stack-md">
            {/* Step 1 */}
            <div className="border border-secondary-container p-8 bg-white">
              <div className="text-[48px] font-bold text-primary mb-4">01</div>
              <h3 className="text-headline-md font-semibold mb-4">Connect</h3>
              <p className="text-body-md text-on-surface-variant leading-relaxed">
                Put on the Muse S headband. Eleven automatically connects via Bluetooth and begins
                reading your brain's electrical activity from 4 EEG sensors at 256 samples per second.
              </p>
            </div>

            {/* Step 2 */}
            <div className="border border-secondary-container p-8 bg-white">
              <div className="text-[48px] font-bold text-primary mb-4">02</div>
              <h3 className="text-headline-md font-semibold mb-4">Calibrate</h3>
              <p className="text-body-md text-on-surface-variant leading-relaxed">
                Spend 2-3 minutes training the system to recognize your unique neural patterns.
                Focus when prompted. Relax when prompted. Clench your jaw to confirm. That's it.
              </p>
            </div>

            {/* Step 3 */}
            <div className="border border-secondary-container p-8 bg-white">
              <div className="text-[48px] font-bold text-primary mb-4">03</div>
              <h3 className="text-headline-md font-semibold mb-4">Communicate</h3>
              <p className="text-body-md text-on-surface-variant leading-relaxed">
                Select phrases from the board using attention and confirm with a jaw clench.
                Your message is spoken aloud instantly. Emergency help is always one action away.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Technology Section */}
      <section className="py-stack-lg px-margin-page bg-white border-t border-secondary-container">
        <div className="max-w-[900px] mx-auto">
          <p className="text-label-lg uppercase tracking-widest text-on-surface-variant mb-4">The Technology</p>
          <h2 className="text-headline-lg md:text-[40px] font-bold tracking-tight mb-stack-md">
            What we detect and how.
          </h2>

          <div className="space-y-8">
            {/* Signal Types */}
            <div className="grid md:grid-cols-2 gap-8">
              <div className="border-l-2 border-primary pl-6">
                <h3 className="text-headline-md font-semibold mb-2">Jaw Clench Detection</h3>
                <p className="text-body-md text-on-surface-variant">
                  EMG artifacts from jaw muscles create distinctive high-frequency bursts that we detect
                  with 95%+ accuracy. Single clench = SELECT. Double clench = YES.
                </p>
              </div>
              <div className="border-l-2 border-primary pl-6">
                <h3 className="text-headline-md font-semibold mb-2">Alpha Wave Attention</h3>
                <p className="text-body-md text-on-surface-variant">
                  When you focus on a target, alpha waves (8-13 Hz) decrease in the frontal cortex.
                  We use this "alpha suppression" to detect where your attention is directed.
                </p>
              </div>
              <div className="border-l-2 border-primary pl-6">
                <h3 className="text-headline-md font-semibold mb-2">Relaxation States</h3>
                <p className="text-body-md text-on-surface-variant">
                  Increased alpha and theta waves indicate relaxation. We track this to prevent
                  fatigue-induced errors and suggest breaks when cognitive load is high.
                </p>
              </div>
              <div className="border-l-2 border-primary pl-6">
                <h3 className="text-headline-md font-semibold mb-2">Engagement Monitoring</h3>
                <p className="text-body-md text-on-surface-variant">
                  Beta wave activity (13-30 Hz) correlates with mental engagement. We use this to
                  adapt the interface speed and confirm intentional actions.
                </p>
              </div>
            </div>

            {/* Technical Specs */}
            <div className="bg-surface-container p-8 border border-secondary-container">
              <h3 className="text-label-lg uppercase tracking-widest mb-4">Technical Specifications</h3>
              <div className="grid md:grid-cols-4 gap-6 text-center">
                <div>
                  <div className="text-[32px] font-bold text-primary">4</div>
                  <div className="text-label-sm text-on-surface-variant uppercase">EEG Channels</div>
                  <div className="text-body-md">TP9, AF7, AF8, TP10</div>
                </div>
                <div>
                  <div className="text-[32px] font-bold text-primary">256</div>
                  <div className="text-label-sm text-on-surface-variant uppercase">Hz Sample Rate</div>
                  <div className="text-body-md">Real-time streaming</div>
                </div>
                <div>
                  <div className="text-[32px] font-bold text-primary">&lt;200</div>
                  <div className="text-label-sm text-on-surface-variant uppercase">ms Latency</div>
                  <div className="text-body-md">Instant response</div>
                </div>
                <div>
                  <div className="text-[32px] font-bold text-primary">95%+</div>
                  <div className="text-label-sm text-on-surface-variant uppercase">Detection Accuracy</div>
                  <div className="text-body-md">After calibration</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Accessibility Section */}
      <section className="py-stack-lg px-margin-page bg-primary text-on-primary">
        <div className="max-w-[900px] mx-auto text-center">
          <p className="text-label-lg uppercase tracking-widest opacity-70 mb-4">Accessibility First</p>
          <h2 className="text-headline-lg md:text-[40px] font-bold tracking-tight mb-6">
            Designed for those who need it most.
          </h2>
          <p className="text-body-lg opacity-90 leading-relaxed mb-8 max-w-2xl mx-auto">
            Every design decision in Eleven prioritizes accessibility. Large 64px touch targets for
            imprecise input. High-contrast monochromatic colors for visual clarity. Immediate audio
            feedback. Emergency help always accessible. No decorative elements—only what's essential.
          </p>

          <div className="grid md:grid-cols-3 gap-8 text-left mt-stack-md">
            <div>
              <h3 className="text-headline-md font-semibold mb-2">ALS Patients</h3>
              <p className="opacity-80 text-body-md">
                Maintain communication as motor function declines. No fine motor control required.
              </p>
            </div>
            <div>
              <h3 className="text-headline-md font-semibold mb-2">Stroke Survivors</h3>
              <p className="opacity-80 text-body-md">
                Express needs during recovery when speech and movement are impaired.
              </p>
            </div>
            <div>
              <h3 className="text-headline-md font-semibold mb-2">Locked-In Syndrome</h3>
              <p className="opacity-80 text-body-md">
                Provide a communication channel when only cognitive function remains intact.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* What We've Built Section */}
      <section className="py-stack-lg px-margin-page bg-surface-container-lowest border-t border-secondary-container">
        <div className="max-w-[900px] mx-auto">
          <p className="text-label-lg uppercase tracking-widest text-on-surface-variant mb-4">What We've Built</p>
          <h2 className="text-headline-lg md:text-[40px] font-bold tracking-tight mb-stack-md">
            A complete communication system.
          </h2>

          <div className="space-y-6">
            <div className="flex gap-6 items-start border-b border-secondary-container pb-6">
              <span className="material-symbols-outlined text-primary text-[32px]">sensors</span>
              <div>
                <h3 className="text-headline-md font-semibold mb-2">Real-Time EEG Processing</h3>
                <p className="text-body-md text-on-surface-variant">
                  BrainFlow-powered signal acquisition with band-pass filtering, artifact rejection,
                  and spectral analysis. Processes 256 samples per second with sub-200ms latency.
                </p>
              </div>
            </div>

            <div className="flex gap-6 items-start border-b border-secondary-container pb-6">
              <span className="material-symbols-outlined text-primary text-[32px]">psychology</span>
              <div>
                <h3 className="text-headline-md font-semibold mb-2">Brain State Classification</h3>
                <p className="text-body-md text-on-surface-variant">
                  Machine learning models that classify engagement, focus, relaxation, and cognitive load.
                  Personalized to each user through quick calibration.
                </p>
              </div>
            </div>

            <div className="flex gap-6 items-start border-b border-secondary-container pb-6">
              <span className="material-symbols-outlined text-primary text-[32px]">chat</span>
              <div>
                <h3 className="text-headline-md font-semibold mb-2">Phrase-Based Communication</h3>
                <p className="text-body-md text-on-surface-variant">
                  Pre-built phrase boards for common needs: basic responses, physical needs, emotions,
                  and emergency alerts. Customizable vocabulary for each user.
                </p>
              </div>
            </div>

            <div className="flex gap-6 items-start">
              <span className="material-symbols-outlined text-primary text-[32px]">record_voice_over</span>
              <div>
                <h3 className="text-headline-md font-semibold mb-2">Instant Text-to-Speech</h3>
                <p className="text-body-md text-on-surface-variant">
                  Selected phrases are immediately spoken aloud using the Web Speech API.
                  Your voice, restored through technology.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-stack-lg px-margin-page bg-white border-t border-secondary-container">
        <div className="max-w-[700px] mx-auto text-center">
          <h2 className="text-headline-lg md:text-[40px] font-bold tracking-tight mb-6">
            Ready to begin?
          </h2>
          <p className="text-body-lg text-on-surface-variant mb-8">
            Connect your Muse S headband and start communicating in minutes.
          </p>
          <Link
            to="/connect"
            className="inline-flex items-center justify-center min-w-[250px] min-h-target-min bg-primary text-on-primary text-label-lg px-10 py-5 transition-all hover:opacity-90 active:scale-95 border border-primary font-semibold"
          >
            Connect Device
            <span className="material-symbols-outlined ml-2" style={{ fontSize: '20px' }}>
              arrow_forward
            </span>
          </Link>
        </div>
      </section>
    </Layout>
  );
}
