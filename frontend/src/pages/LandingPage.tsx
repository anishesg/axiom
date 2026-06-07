import { Link } from 'react-router-dom';
import { Layout } from '../components/layout/Layout';
import { SignalIndicator } from '../components/ui/SignalIndicator';
import elevenLogo from '../assets/elevenlogo.png';
import infographic from '../assets/infographic.png';

export function LandingPage() {
  return (
    <Layout>
      {/* Hero Section */}
      <section className="min-h-[85vh] flex flex-col items-center justify-center bg-background px-margin-page py-stack-lg">
        <div className="max-w-[860px] mx-auto text-center">
          {/* Logo */}
          <div className="mb-8 animate-fade-in">
            <img
              src={elevenLogo}
              alt="Eleven"
              className="h-56 md:h-72 lg:h-80 mx-auto object-contain"
            />
          </div>

          {/* Headline */}
          <h1 className="text-display md:text-[56px] lg:text-[64px] mb-6 tracking-tight font-semibold leading-[1.08] animate-fade-in stagger-1">
            Communication through thought.
          </h1>

          {/* Subtitle */}
          <p className="text-body-lg md:text-[19px] text-on-surface-variant max-w-[580px] mx-auto mb-10 leading-relaxed animate-fade-in stagger-2">
            A brain-computer interface that enables people with severe motor disabilities
            to communicate using only their thoughts.
          </p>

          {/* CTAs */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 animate-fade-in stagger-3">
            <Link
              to="/connect"
              className="w-full sm:w-auto min-w-[180px] bg-primary text-on-primary text-[14px] font-medium px-7 py-3.5 rounded-lg transition-all hover:shadow-lg hover:-translate-y-0.5 active:translate-y-0 active:shadow-md flex items-center justify-center gap-2"
            >
              Get Started
              <span className="material-symbols-outlined text-[18px]">arrow_forward</span>
            </Link>
            <Link
              to="/research"
              className="text-[14px] font-medium text-on-surface-variant flex items-center gap-2 px-6 py-3.5 rounded-lg hover:bg-surface-container transition-all"
            >
              Learn more
              <span className="material-symbols-outlined text-[16px]">arrow_forward</span>
            </Link>
          </div>

          {/* Signal Status */}
          <div className="mt-16 animate-fade-in stagger-4">
            <SignalIndicator status="ready" label="System Ready" />
          </div>
        </div>
      </section>

      {/* Mission Section */}
      <section className="py-20 px-margin-page bg-background">
        <div className="max-w-[720px] mx-auto">
          <p className="text-[11px] font-medium uppercase tracking-[0.15em] text-on-surface-variant mb-4">Our Mission</p>
          <h2 className="text-headline-lg md:text-[36px] font-semibold tracking-tight mb-6 leading-tight">
            Restoring voice to those who have lost it.
          </h2>
          <div className="flex gap-6">
            <div className="flex-1 space-y-5 text-body-md text-on-surface-variant leading-relaxed">
              <p>
                Every year, thousands of people lose the ability to speak due to ALS, stroke, traumatic brain injury,
                or other conditions. Current assistive technologies require precise motor control that many patients
                simply don't have.
              </p>
              <p>
                Eleven changes this. Using a consumer-grade EEG headband (Muse S), we detect neural signals
                associated with deliberate mental actions like jaw clenches, focused attention, and relaxation states,
                then translate them into communication.
              </p>
            </div>
            <img
              src={infographic}
              alt="EEG System Diagram"
              className="w-[350px] h-auto object-contain flex-shrink-0 hidden sm:block"
            />
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section id="how-it-works" className="py-20 px-margin-page bg-surface-container-lowest">
        <div className="max-w-[1000px] mx-auto">
          <div className="text-center mb-14">
            <p className="text-[11px] font-medium uppercase tracking-[0.15em] text-on-surface-variant mb-4">How It Works</p>
            <h2 className="text-headline-lg md:text-[36px] font-semibold tracking-tight leading-tight">
              Three simple steps to communicate.
            </h2>
          </div>

          <div className="grid md:grid-cols-3 gap-6">
            {/* Step 1 */}
            <div className="bg-white border border-outline-variant rounded-xl p-7 hover:shadow-md hover:-translate-y-1 transition-all duration-300">
              <div className="text-[11px] font-medium uppercase tracking-[0.15em] text-on-surface-variant mb-5">Step 01</div>
              <h3 className="text-headline-md font-medium mb-3">Connect</h3>
              <p className="text-body-md text-on-surface-variant leading-relaxed">
                Put on the Muse S headband. Eleven connects via Bluetooth and begins
                reading your brain's electrical activity from 4 EEG sensors.
              </p>
            </div>

            {/* Step 2 */}
            <div className="bg-white border border-outline-variant rounded-xl p-7 hover:shadow-md hover:-translate-y-1 transition-all duration-300">
              <div className="text-[11px] font-medium uppercase tracking-[0.15em] text-on-surface-variant mb-5">Step 02</div>
              <h3 className="text-headline-md font-medium mb-3">Calibrate</h3>
              <p className="text-body-md text-on-surface-variant leading-relaxed">
                Spend 2-3 minutes training the system to recognize your unique neural patterns.
                Focus when prompted. Relax when prompted.
              </p>
            </div>

            {/* Step 3 */}
            <div className="bg-white border border-outline-variant rounded-xl p-7 hover:shadow-md hover:-translate-y-1 transition-all duration-300">
              <div className="text-[11px] font-medium uppercase tracking-[0.15em] text-on-surface-variant mb-5">Step 03</div>
              <h3 className="text-headline-md font-medium mb-3">Communicate</h3>
              <p className="text-body-md text-on-surface-variant leading-relaxed">
                Select phrases using attention and confirm with a jaw clench.
                Your message is spoken aloud instantly.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Accessibility Section */}
      <section className="py-20 px-margin-page bg-primary-container text-white">
        <div className="max-w-[860px] mx-auto">
          <div className="text-center mb-12">
            <p className="text-[11px] font-medium uppercase tracking-[0.15em] text-white/60 mb-4">Accessibility First</p>
            <h2 className="text-headline-lg md:text-[36px] font-semibold tracking-tight mb-5 leading-tight">
              Designed for those who need it most.
            </h2>
            <p className="text-body-md text-white/80 leading-relaxed max-w-[560px] mx-auto">
              Every design decision prioritizes accessibility. Large touch targets, high-contrast colors,
              immediate audio feedback. No decorative elements, only what's essential.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-6">
            <div className="bg-white/5 rounded-xl p-6 backdrop-blur-sm border border-white/10">
              <h3 className="text-headline-md font-medium mb-2">ALS Patients</h3>
              <p className="text-white/70 text-body-md leading-relaxed">
                Maintain communication as motor function declines. No fine motor control required.
              </p>
            </div>
            <div className="bg-white/5 rounded-xl p-6 backdrop-blur-sm border border-white/10">
              <h3 className="text-headline-md font-medium mb-2">Stroke Survivors</h3>
              <p className="text-white/70 text-body-md leading-relaxed">
                Express needs during recovery when speech and movement are impaired.
              </p>
            </div>
            <div className="bg-white/5 rounded-xl p-6 backdrop-blur-sm border border-white/10">
              <h3 className="text-headline-md font-medium mb-2">Locked-In Syndrome</h3>
              <p className="text-white/70 text-body-md leading-relaxed">
                A communication channel when only cognitive function remains intact.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 px-margin-page bg-surface-container-lowest">
        <div className="max-w-[560px] mx-auto text-center">
          <h2 className="text-headline-lg md:text-[36px] font-semibold tracking-tight mb-5">
            Ready to begin?
          </h2>
          <p className="text-body-md text-on-surface-variant mb-8">
            Connect your Muse S headband and start communicating in minutes.
          </p>
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Link
              to="/connect"
              className="inline-flex items-center justify-center bg-primary text-on-primary text-[14px] font-medium px-8 py-4 rounded-lg transition-all hover:shadow-lg hover:-translate-y-0.5 active:translate-y-0"
            >
              Connect Device
              <span className="material-symbols-outlined ml-2 text-[18px]">arrow_forward</span>
            </Link>
            <Link
              to="/team"
              className="inline-flex items-center justify-center text-[14px] font-medium text-on-surface-variant px-6 py-4 rounded-lg transition-all hover:bg-surface-container-low"
            >
              Meet the Team
              <span className="material-symbols-outlined ml-2 text-[16px]">group</span>
            </Link>
          </div>
        </div>
      </section>
    </Layout>
  );
}
