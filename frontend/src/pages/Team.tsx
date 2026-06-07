import { Layout } from '../components/layout/Layout';
import krishPhoto from '../assets/krish.png';
import anishPhoto from '../assets/anish.png';
import dhruvaPhoto from '../assets/dhruva.png';
import yougiPhoto from '../assets/yougi.png';

interface TeamMember {
  name: string;
  photo: string;
  imagePosition?: string;
}

const teamMembers: TeamMember[] = [
  { name: 'Krish Malik', photo: krishPhoto },
  { name: 'Anish Kataria', photo: anishPhoto },
  { name: 'Dhruva Uppaluri', photo: dhruvaPhoto, imagePosition: 'center 30%' },
  { name: 'Yougi Jain', photo: yougiPhoto },
];

export function Team() {
  return (
    <Layout showBackButton backTo="/" backLabel="Home">
      <main className="flex-grow px-margin-page py-16">
        <div className="max-w-[900px] mx-auto">
          {/* Header */}
          <div className="text-center mb-16">
            <p className="text-[11px] font-medium uppercase tracking-[0.15em] text-on-surface-variant mb-4">
              The People Behind Eleven
            </p>
            <h1 className="text-headline-lg md:text-[42px] font-semibold tracking-tight mb-4">
              Our Team
            </h1>
            <p className="text-body-md text-on-surface-variant max-w-[480px] mx-auto leading-relaxed">
              A passionate group of engineers and researchers dedicated to making communication accessible for everyone.
            </p>
          </div>

          {/* Team Grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 md:gap-8">
            {teamMembers.map((member, index) => (
              <div
                key={member.name}
                className="group animate-fade-in"
                style={{ animationDelay: `${index * 0.1}s` }}
              >
                <div className="aspect-square rounded-2xl overflow-hidden bg-surface-container-low mb-4 border border-outline-variant/30 transition-all duration-300 group-hover:shadow-lg group-hover:-translate-y-1">
                  <img
                    src={member.photo}
                    alt={member.name}
                    className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
                    style={member.imagePosition ? { objectPosition: member.imagePosition } : undefined}
                  />
                </div>
                <h3 className="text-[15px] font-medium text-center text-on-surface">
                  {member.name}
                </h3>
              </div>
            ))}
          </div>
        </div>
      </main>
    </Layout>
  );
}
