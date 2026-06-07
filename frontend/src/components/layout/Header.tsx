import { Link, useLocation } from 'react-router-dom';

interface HeaderProps {
  showNav?: boolean;
  showBackButton?: boolean;
  backTo?: string;
  backLabel?: string;
}

export function Header({
  showNav = true,
  showBackButton = false,
  backTo = "/",
  backLabel = "Back"
}: HeaderProps) {
  const location = useLocation();
  const isHome = location.pathname === '/';

  return (
    <header className="w-full top-0 bg-background border-b border-outline-variant/30 z-50">
      <div className="flex justify-between items-center w-full px-margin-page py-4 max-w-[1200px] mx-auto">
        {/* Left side */}
        <div className="flex items-center gap-4">
          {showBackButton ? (
            <Link
              to={backTo}
              className="flex items-center gap-2 text-on-surface-variant hover:text-on-surface transition-colors"
            >
              <span className="material-symbols-outlined text-[20px]">arrow_back</span>
              <span className="text-[14px] font-medium">{backLabel}</span>
            </Link>
          ) : (
            <Link to="/" className="text-[20px] md:text-[22px] font-extralight tracking-[0.15em] text-on-surface hover:opacity-60 transition-opacity">
              eleven
            </Link>
          )}
        </div>

        {/* Navigation */}
        <nav className="flex items-center gap-8">
          <div className="flex gap-6 items-center">
            <Link to="/" className="text-[14px] font-medium text-primary hover:opacity-80 transition-opacity">Home</Link>
            <Link to="/research" className="text-[14px] font-medium text-on-surface-variant hover:text-on-surface transition-colors">Research</Link>
            <Link to="/team" className="text-[14px] font-medium text-on-surface-variant hover:text-on-surface transition-colors">Team</Link>
          </div>
          <div className="flex gap-3 items-center">
            <Link to="/connect" className="material-symbols-outlined text-[20px] text-on-surface-variant hover:text-primary transition-colors">
              sensors
            </Link>
            <Link to="/settings" className="material-symbols-outlined text-[20px] text-on-surface-variant hover:text-primary transition-colors">
              settings
            </Link>
          </div>
        </nav>
      </div>
    </header>
  );
}
