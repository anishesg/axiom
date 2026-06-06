import { Link, useLocation } from 'react-router-dom';

// ELEVEN wordmark as base64 (from design files)
const ELEVEN_LOGO = "data:image/png;base64,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";

interface HeaderProps {
  showNav?: boolean;
  showBackButton?: boolean;
  backTo?: string;
  backLabel?: string;
}

export function Header({
  showNav = true,
  showBackButton = false,
  backTo = "/communicate",
  backLabel = "Back to Chat"
}: HeaderProps) {
  const location = useLocation();
  const isHome = location.pathname === '/';

  return (
    <header className="w-full top-0 bg-background border-b border-on-background z-50">
      <div className="flex justify-between items-center w-full px-margin-page py-4 max-w-[1200px] mx-auto">
        {/* Left side */}
        <div className="flex items-center gap-4">
          {showBackButton ? (
            <Link
              to={backTo}
              className="flex items-center gap-2 text-on-surface-variant hover:opacity-80 transition-opacity active:scale-95"
            >
              <span className="material-symbols-outlined text-[20px]">arrow_back</span>
              <span className="text-label-lg">{backLabel}</span>
            </Link>
          ) : isHome ? (
            <span className="text-display tracking-tighter text-primary">ELEVEN</span>
          ) : (
            <Link to="/">
              <img
                src={ELEVEN_LOGO}
                alt="ELEVEN"
                className="h-6 md:h-8 object-contain"
              />
            </Link>
          )}
        </div>

        {/* Navigation (desktop) */}
        {showNav && isHome && (
          <nav className="hidden md:flex gap-stack-md items-center">
            <a href="#" className="text-primary font-bold hover:opacity-80 transition-opacity">Home</a>
            <a href="#" className="text-on-surface-variant hover:opacity-80 transition-opacity">Technology</a>
            <a href="#" className="text-on-surface-variant hover:opacity-80 transition-opacity">Research</a>
            <div className="flex gap-4 ml-4">
              <Link to="/connect" className="material-symbols-outlined text-primary cursor-pointer hover:opacity-80">
                sensors
              </Link>
              <Link to="/settings" className="material-symbols-outlined text-primary cursor-pointer hover:opacity-80">
                settings
              </Link>
            </div>
          </nav>
        )}

        {/* Right side icons (non-home pages) */}
        {!isHome && (
          <div className="flex items-center gap-4">
            <Link to="/connect" className="material-symbols-outlined text-primary p-2 hover:opacity-80">
              sensors
            </Link>
            <Link to="/settings" className="material-symbols-outlined text-primary p-2 hover:opacity-80">
              settings
            </Link>
          </div>
        )}

        {/* Mobile menu button */}
        {showNav && isHome && (
          <button className="md:hidden material-symbols-outlined text-primary">
            menu
          </button>
        )}
      </div>
    </header>
  );
}
