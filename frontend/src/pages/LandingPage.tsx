import { Link } from 'react-router-dom';
import { Layout } from '../components/layout/Layout';
import { SignalIndicator } from '../components/ui/SignalIndicator';

// Placeholder images for the bento grid
const HERO_IMAGE_1 = 'https://images.unsplash.com/photo-1559757175-5700dde675bc?w=800&h=600&fit=crop';
const HERO_IMAGE_2 = 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400&h=600&fit=crop';

export function LandingPage() {
  return (
    <Layout fullHeight>
      <main className="min-h-screen flex flex-col items-center justify-center bg-surface-container-lowest">
        {/* Hero Section */}
        <section className="max-w-[1200px] mx-auto px-margin-page w-full flex flex-col items-center text-center py-stack-lg">
          {/* Logo Mark */}
          <div className="mb-stack-md">
            <div className="h-12 md:h-16 flex items-center justify-center">
              <span className="text-display text-primary font-bold tracking-tighter">XI</span>
            </div>
          </div>

          {/* Headline */}
          <h1 className="text-display md:text-[64px] mb-4 tracking-tighter max-w-2xl font-bold leading-[1.1]">
            Communication through thought.
          </h1>

          {/* Subtitle */}
          <p className="text-body-lg md:text-[24px] text-on-surface-variant max-w-xl mb-stack-md leading-relaxed">
            A brain-computer interface designed for accessibility and clarity.
          </p>

          {/* CTAs */}
          <div className="flex flex-col md:flex-row items-center gap-stack-sm w-full md:w-auto">
            <Link
              to="/connect"
              className="w-full md:w-auto min-w-[200px] min-h-target-min bg-primary text-on-primary text-label-lg px-8 py-4 transition-all hover:opacity-90 active:scale-95 border border-primary flex items-center justify-center font-semibold"
            >
              Get Started
            </Link>
            <Link
              to="/technology"
              className="text-label-lg text-primary flex items-center gap-2 px-8 py-4 hover:underline transition-all font-semibold"
            >
              Learn more
              <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>
                arrow_forward
              </span>
            </Link>
          </div>

          {/* Signal Status Indicator */}
          <div className="mt-stack-lg">
            <SignalIndicator status="ready" label="Signal Status: Ready" />
          </div>
        </section>

        {/* Bento Grid Section */}
        <section className="max-w-[1200px] mx-auto px-margin-page w-full pb-stack-lg">
          <div className="grid grid-cols-1 md:grid-cols-12 gap-gutter">
            {/* Large Image */}
            <div className="md:col-span-8 h-[400px] border border-surface-variant relative overflow-hidden group">
              <img
                src={HERO_IMAGE_1}
                alt="The Eleven Interface - minimalist brain-computer interface wearable device"
                className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-105"
              />
              <div className="absolute bottom-0 left-0 p-stack-sm bg-surface-container-lowest/90 backdrop-blur-sm border-t border-r border-surface-variant">
                <p className="text-label-sm text-primary font-medium">01 // THE INTERFACE</p>
              </div>
            </div>

            {/* Small Image */}
            <div className="md:col-span-4 h-[400px] border border-surface-variant relative overflow-hidden group">
              <img
                src={HERO_IMAGE_2}
                alt="Neural Flow visualization - conceptual neural activity"
                className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-105"
              />
              <div className="absolute bottom-0 left-0 p-stack-sm bg-surface-container-lowest/90 backdrop-blur-sm border-t border-r border-surface-variant">
                <p className="text-label-sm text-primary font-medium">02 // NEURAL FLOW</p>
              </div>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}
