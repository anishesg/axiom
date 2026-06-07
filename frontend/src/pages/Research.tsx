import { Layout } from '../components/layout/Layout';

export function Research() {
  return (
    <Layout showBackButton backTo="/" backLabel="Home">
      <main className="flex-grow px-margin-page py-16">
        <div className="max-w-[900px] mx-auto">
          {/* Header */}
          <div className="mb-16">
            <p className="text-[11px] font-medium uppercase tracking-[0.15em] text-on-surface-variant mb-4">
              Research & Documentation
            </p>
            <h1 className="text-headline-lg md:text-[42px] font-semibold tracking-tight mb-6">
              The Science Behind Eleven
            </h1>
            <p className="text-body-lg text-on-surface-variant max-w-[700px] leading-relaxed">
              Eleven is built on decades of neuroscience research and modern signal processing techniques.
              This page documents our approach, the challenges we faced, and how we solved them.
            </p>
          </div>

          {/* The Problem */}
          <section className="mb-16">
            <h2 className="text-headline-md md:text-[28px] font-semibold tracking-tight mb-6">
              The Problem We're Solving
            </h2>
            <div className="space-y-5 text-body-md text-on-surface-variant leading-relaxed">
              <p>
                Every year, over 140,000 people in the United States alone are diagnosed with conditions
                that severely limit their ability to communicate. ALS progressively destroys motor neurons,
                leaving patients unable to speak, write, or gesture. Stroke survivors often experience
                aphasia or paralysis that makes traditional communication impossible. Patients with
                locked-in syndrome retain full cognitive function but cannot move any muscles except,
                in some cases, their eyes.
              </p>
              <p>
                Current assistive technologies like eye trackers and switch-based systems require
                precise motor control that many patients simply do not have. Eye trackers fail when
                patients lose voluntary eye movement. Switch systems require consistent muscle activation
                that deteriorates as conditions progress. These solutions also tend to be expensive,
                costing thousands of dollars, putting them out of reach for many families.
              </p>
              <p>
                We asked a simple question: what if we could bypass motor function entirely and
                communicate directly through brain signals?
              </p>
            </div>
          </section>

          {/* Our Approach */}
          <section className="mb-16">
            <h2 className="text-headline-md md:text-[28px] font-semibold tracking-tight mb-6">
              Our Approach
            </h2>
            <div className="space-y-5 text-body-md text-on-surface-variant leading-relaxed">
              <p>
                Eleven uses electroencephalography (EEG) to detect electrical activity in the brain.
                When neurons fire, they create tiny electrical signals that propagate to the scalp
                where they can be measured by electrodes. By analyzing patterns in these signals,
                we can infer mental states and detect deliberate actions.
              </p>
              <p>
                We chose to build on the Muse S, a consumer-grade EEG headband originally designed
                for meditation. While medical-grade EEG systems can cost over $10,000 and require
                trained technicians to operate, the Muse S costs under $400 and can be set up by
                anyone in minutes. It provides four dry electrodes positioned at TP9, AF7, AF8,
                and TP10, sampling at 256 Hz.
              </p>
              <p>
                The tradeoff is signal quality. Consumer EEG has more noise, fewer channels, and
                lower spatial resolution than clinical systems. Our challenge was to extract
                reliable communication signals from this limited data.
              </p>
            </div>
          </section>

          {/* Signal Processing */}
          <section className="mb-16">
            <h2 className="text-headline-md md:text-[28px] font-semibold tracking-tight mb-6">
              Signal Processing Pipeline
            </h2>
            <div className="bg-surface-container-low rounded-xl p-8 mb-8">
              <div className="grid md:grid-cols-4 gap-6 text-center">
                <div>
                  <div className="text-[11px] text-on-surface-variant uppercase tracking-wide mb-2">Step 1</div>
                  <div className="text-headline-md font-medium mb-1">Acquisition</div>
                  <p className="text-[13px] text-on-surface-variant">BrainFlow SDK streams raw EEG at 256 Hz</p>
                </div>
                <div>
                  <div className="text-[11px] text-on-surface-variant uppercase tracking-wide mb-2">Step 2</div>
                  <div className="text-headline-md font-medium mb-1">Filtering</div>
                  <p className="text-[13px] text-on-surface-variant">Bandpass filter removes noise outside 1-50 Hz</p>
                </div>
                <div>
                  <div className="text-[11px] text-on-surface-variant uppercase tracking-wide mb-2">Step 3</div>
                  <div className="text-headline-md font-medium mb-1">Extraction</div>
                  <p className="text-[13px] text-on-surface-variant">FFT computes power in each frequency band</p>
                </div>
                <div>
                  <div className="text-[11px] text-on-surface-variant uppercase tracking-wide mb-2">Step 4</div>
                  <div className="text-headline-md font-medium mb-1">Classification</div>
                  <p className="text-[13px] text-on-surface-variant">ML models detect states and actions</p>
                </div>
              </div>
            </div>
            <div className="space-y-5 text-body-md text-on-surface-variant leading-relaxed">
              <p>
                Raw EEG signals are noisy. Power line interference at 60 Hz, muscle artifacts from
                facial movements, and electrode drift all contaminate the data. Our first processing
                step applies a bandpass filter between 1 and 50 Hz, removing low-frequency drift
                and high-frequency noise while preserving the brain rhythms we care about.
              </p>
              <p>
                We then compute the Fast Fourier Transform (FFT) on sliding windows of data to
                extract power spectral density. This tells us how much energy exists in each
                frequency band: delta (1-4 Hz), theta (4-8 Hz), alpha (8-13 Hz), beta (13-30 Hz),
                and gamma (30-50 Hz). Each band correlates with different mental states.
              </p>
            </div>
          </section>

          {/* Brain Rhythms */}
          <section className="mb-16">
            <h2 className="text-headline-md md:text-[28px] font-semibold tracking-tight mb-6">
              Understanding Brain Rhythms
            </h2>
            <div className="grid md:grid-cols-2 gap-6 mb-8">
              <div className="border border-outline-variant rounded-xl p-6">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-3 h-3 rounded-full bg-blue-500"></div>
                  <h3 className="text-headline-md font-medium">Alpha Waves (8-13 Hz)</h3>
                </div>
                <p className="text-body-md text-on-surface-variant leading-relaxed">
                  Alpha waves are prominent when you close your eyes and relax. They decrease
                  when you focus on something specific, a phenomenon called alpha suppression
                  or event-related desynchronization (ERD). We use this to detect directed attention.
                </p>
              </div>
              <div className="border border-outline-variant rounded-xl p-6">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-3 h-3 rounded-full bg-orange-500"></div>
                  <h3 className="text-headline-md font-medium">Beta Waves (13-30 Hz)</h3>
                </div>
                <p className="text-body-md text-on-surface-variant leading-relaxed">
                  Beta activity increases during active thinking, problem-solving, and concentration.
                  Higher beta power indicates mental engagement. We use the ratio of beta to alpha
                  as an engagement metric to adapt interface timing.
                </p>
              </div>
              <div className="border border-outline-variant rounded-xl p-6">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-3 h-3 rounded-full bg-purple-500"></div>
                  <h3 className="text-headline-md font-medium">Theta Waves (4-8 Hz)</h3>
                </div>
                <p className="text-body-md text-on-surface-variant leading-relaxed">
                  Theta waves appear during drowsiness, light sleep, and deep relaxation.
                  Elevated theta combined with decreased beta indicates fatigue. We monitor
                  this to suggest breaks and prevent errors from tiredness.
                </p>
              </div>
              <div className="border border-outline-variant rounded-xl p-6">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-3 h-3 rounded-full bg-red-500"></div>
                  <h3 className="text-headline-md font-medium">EMG Artifacts</h3>
                </div>
                <p className="text-body-md text-on-surface-variant leading-relaxed">
                  When you clench your jaw, the temporalis muscles create high-frequency
                  electrical bursts that appear in the EEG. Rather than filtering these out,
                  we detect them as deliberate control signals with over 95% accuracy.
                </p>
              </div>
            </div>
          </section>

          {/* Jaw Clench Detection */}
          <section className="mb-16">
            <h2 className="text-headline-md md:text-[28px] font-semibold tracking-tight mb-6">
              Jaw Clench Detection
            </h2>
            <div className="space-y-5 text-body-md text-on-surface-variant leading-relaxed">
              <p>
                The jaw clench is our primary selection mechanism. When the temporalis and masseter
                muscles contract during a clench, they produce a distinctive electromyographic (EMG)
                signal that appears as a high-amplitude, high-frequency burst in the EEG channels
                near the temples (TP9 and TP10).
              </p>
              <p>
                Our detection algorithm computes the root mean square (RMS) amplitude in a sliding
                window and compares it against a personalized threshold established during calibration.
                We also analyze the frequency content: jaw clenches produce energy primarily above
                20 Hz, while true brain signals are concentrated below 30 Hz. This frequency signature
                helps us distinguish intentional clenches from other artifacts.
              </p>
              <p>
                To prevent false positives from teeth grinding or facial tension, we require the
                signal to exceed the threshold for a minimum duration (typically 200ms) and then
                return to baseline within a maximum duration (typically 1 second). This temporal
                pattern matching rejects noise while accepting deliberate clenches.
              </p>
            </div>
          </section>

          {/* Calibration */}
          <section className="mb-16">
            <h2 className="text-headline-md md:text-[28px] font-semibold tracking-tight mb-6">
              Personalized Calibration
            </h2>
            <div className="space-y-5 text-body-md text-on-surface-variant leading-relaxed">
              <p>
                Every brain is different. The amplitude, frequency distribution, and spatial
                pattern of EEG signals vary significantly between individuals. A threshold that
                works perfectly for one person might produce constant false positives for another
                or never trigger at all.
              </p>
              <p>
                Our calibration process collects baseline data during relaxation and samples of
                deliberate actions like jaw clenches and focused attention. We use this data to
                establish personalized thresholds for each signal type. The process takes 2-3
                minutes and significantly improves detection accuracy.
              </p>
              <p>
                During calibration, we also compute the user's baseline alpha power and beta/alpha
                ratio. These baselines let us detect relative changes in attention and engagement
                rather than relying on absolute values that vary between sessions and individuals.
              </p>
            </div>
          </section>

          {/* Cognitive State Detection */}
          <section className="mb-16">
            <h2 className="text-headline-md md:text-[28px] font-semibold tracking-tight mb-6">
              Cognitive State Detection
            </h2>
            <div className="space-y-5 text-body-md text-on-surface-variant leading-relaxed">
              <p>
                Beyond discrete actions like jaw clenches, we continuously monitor cognitive state
                to adapt the interface and enable more natural communication. Our state detection
                system estimates four primary dimensions:
              </p>
            </div>
            <div className="grid md:grid-cols-2 gap-4 mt-6">
              <div className="bg-surface-container rounded-lg p-5">
                <h4 className="font-medium mb-2">Engagement</h4>
                <p className="text-[14px] text-on-surface-variant">
                  Computed from beta/alpha ratio. High engagement indicates active participation;
                  low engagement might mean distraction or fatigue.
                </p>
              </div>
              <div className="bg-surface-container rounded-lg p-5">
                <h4 className="font-medium mb-2">Focus</h4>
                <p className="text-[14px] text-on-surface-variant">
                  Measured by alpha suppression in frontal channels. Sustained focus triggers
                  selection in our attention-based interface.
                </p>
              </div>
              <div className="bg-surface-container rounded-lg p-5">
                <h4 className="font-medium mb-2">Relaxation</h4>
                <p className="text-[14px] text-on-surface-variant">
                  Indicated by elevated alpha and theta power. We track this to detect rest
                  states and prevent accidental activations.
                </p>
              </div>
              <div className="bg-surface-container rounded-lg p-5">
                <h4 className="font-medium mb-2">Valence</h4>
                <p className="text-[14px] text-on-surface-variant">
                  Estimated from frontal asymmetry patterns. Relatively more left frontal
                  activity correlates with positive affect.
                </p>
              </div>
            </div>
          </section>

          {/* The Interface */}
          <section className="mb-16">
            <h2 className="text-headline-md md:text-[28px] font-semibold tracking-tight mb-6">
              Communication Interface Design
            </h2>
            <div className="space-y-5 text-body-md text-on-surface-variant leading-relaxed">
              <p>
                Traditional brain-computer interfaces often use complex paradigms like P300
                spellers that require users to watch flashing letters for minutes to spell a
                single word. We designed Eleven for speed and simplicity.
              </p>
              <p>
                Our phrase board presents common messages organized by category: basic responses
                (yes, no, maybe), physical needs (water, pain, bathroom), emotions (happy, sad,
                tired), and emergencies (help, call nurse). Users navigate by focusing attention
                on their desired category, then focusing on the specific phrase. A jaw clench
                confirms selection.
              </p>
              <p>
                This hybrid approach combines attention-based navigation with motor confirmation.
                The attention component is inherently noisy, but the jaw clench provides a clean,
                high-confidence selection signal. Together, they enable reliable communication
                at practical speeds.
              </p>
            </div>
          </section>

          {/* AI Integration */}
          <section className="mb-16">
            <h2 className="text-headline-md md:text-[28px] font-semibold tracking-tight mb-6">
              AI-Assisted Communication
            </h2>
            <div className="space-y-5 text-body-md text-on-surface-variant leading-relaxed">
              <p>
                For our Conversation Mode, we integrated large language models to enable more
                natural dialogue. When someone speaks to the EEG user, their speech is transcribed
                and the system enters a listening period where it monitors cognitive state.
              </p>
              <p>
                Based on the detected engagement, focus, relaxation, and valence patterns, the AI
                generates a contextually appropriate response. High engagement with positive valence
                might produce an enthusiastic agreement; low engagement with high relaxation might
                indicate the user is tired and wants to rest.
              </p>
              <p>
                This approach enables fluid conversation without requiring the user to select from
                predefined phrases. The AI acts as an interpreter, translating brain states into
                natural language while the user retains agency through the ability to accept,
                reject, or modify suggested responses.
              </p>
            </div>
          </section>

          {/* Technical Architecture */}
          <section className="mb-16">
            <h2 className="text-headline-md md:text-[28px] font-semibold tracking-tight mb-6">
              Technical Architecture
            </h2>
            <div className="bg-surface-container-low rounded-xl p-8 mb-8">
              <h3 className="text-[11px] font-medium uppercase tracking-[0.15em] text-on-surface-variant mb-6">System Components</h3>
              <div className="grid md:grid-cols-3 gap-6">
                <div>
                  <div className="text-headline-md font-medium mb-2">Frontend</div>
                  <p className="text-[14px] text-on-surface-variant mb-3">
                    React with TypeScript, Tailwind CSS for styling, WebSocket for real-time data
                  </p>
                  <ul className="text-[13px] text-on-surface-variant space-y-1">
                    <li>• Real-time EEG visualization</li>
                    <li>• Phrase board interface</li>
                    <li>• Calibration wizard</li>
                    <li>• Settings management</li>
                  </ul>
                </div>
                <div>
                  <div className="text-headline-md font-medium mb-2">Backend</div>
                  <p className="text-[14px] text-on-surface-variant mb-3">
                    Python FastAPI server, BrainFlow for EEG acquisition, NumPy/SciPy for DSP
                  </p>
                  <ul className="text-[13px] text-on-surface-variant space-y-1">
                    <li>• Signal processing pipeline</li>
                    <li>• State classification</li>
                    <li>• WebSocket streaming</li>
                    <li>• LLM integration</li>
                  </ul>
                </div>
                <div>
                  <div className="text-headline-md font-medium mb-2">Hardware</div>
                  <p className="text-[14px] text-on-surface-variant mb-3">
                    Muse S EEG headband with Bluetooth Low Energy connectivity
                  </p>
                  <ul className="text-[13px] text-on-surface-variant space-y-1">
                    <li>• 4 EEG channels</li>
                    <li>• 256 Hz sample rate</li>
                    <li>• Dry electrodes</li>
                    <li>• 10+ hour battery</li>
                  </ul>
                </div>
              </div>
            </div>
          </section>

          {/* Limitations */}
          <section className="mb-16">
            <h2 className="text-headline-md md:text-[28px] font-semibold tracking-tight mb-6">
              Current Limitations
            </h2>
            <div className="space-y-5 text-body-md text-on-surface-variant leading-relaxed">
              <p>
                We believe in being transparent about what our system can and cannot do.
                Consumer EEG has fundamental limitations that affect reliability.
              </p>
              <p>
                Signal quality varies with electrode contact. Hair, sweat, and movement can
                all degrade the signal. Some users may find it difficult to achieve consistent
                contact, especially those with thick or curly hair.
              </p>
              <p>
                Attention-based selection is inherently probabilistic. Unlike a button press,
                attention does not have a clear binary state. Our threshold-based detection
                will occasionally produce false positives and false negatives. The jaw clench
                confirmation helps mitigate this, but the system is not as deterministic as
                traditional input devices.
              </p>
              <p>
                Cognitive state inference from four EEG channels is an approximation. We cannot
                read specific thoughts or emotions with certainty. Our state metrics are
                statistical correlations, not direct measurements. They work well in aggregate
                but may not always reflect the user's actual mental state.
              </p>
            </div>
          </section>

          {/* Future Work */}
          <section className="mb-16">
            <h2 className="text-headline-md md:text-[28px] font-semibold tracking-tight mb-6">
              Future Directions
            </h2>
            <div className="space-y-5 text-body-md text-on-surface-variant leading-relaxed">
              <p>
                We see several paths to improve Eleven. Machine learning models trained on
                larger datasets could improve classification accuracy. Adaptive algorithms
                that update calibration over time could account for changing signal characteristics.
                Integration with additional sensors like accelerometers could help reject
                motion artifacts.
              </p>
              <p>
                On the interface side, we are exploring hierarchical phrase organization that
                could enable faster navigation, predictive text that suggests likely responses
                based on conversation context, and customizable vocabulary that adapts to each
                user's communication patterns.
              </p>
              <p>
                Our ultimate goal is to make brain-computer interface technology accessible to
                everyone who needs it. This means not just technical improvements, but also
                reducing cost, simplifying setup, and building a community of users and developers
                who can contribute to the platform.
              </p>
            </div>
          </section>

          {/* References */}
          <section className="mb-8">
            <h2 className="text-headline-md md:text-[28px] font-semibold tracking-tight mb-6">
              References & Further Reading
            </h2>
            <div className="space-y-4 text-body-md text-on-surface-variant">
              <div className="border-l-2 border-outline-variant pl-4">
                <p className="font-medium text-on-surface">BrainFlow Documentation</p>
                <p className="text-[14px]">Open-source library for biosignal acquisition and processing</p>
              </div>
              <div className="border-l-2 border-outline-variant pl-4">
                <p className="font-medium text-on-surface">Muse S Technical Specifications</p>
                <p className="text-[14px]">Hardware details for the consumer EEG headband</p>
              </div>
              <div className="border-l-2 border-outline-variant pl-4">
                <p className="font-medium text-on-surface">Event-Related Desynchronization (ERD)</p>
                <p className="text-[14px]">Pfurtscheller & Lopes da Silva, 1999. Clinical Neurophysiology</p>
              </div>
              <div className="border-l-2 border-outline-variant pl-4">
                <p className="font-medium text-on-surface">Frontal EEG Asymmetry and Emotion</p>
                <p className="text-[14px]">Davidson, 2004. Biological Psychology</p>
              </div>
            </div>
          </section>
        </div>
      </main>
    </Layout>
  );
}
