export function Footer() {
  return (
    <footer className="w-full bg-background border-t border-secondary-container">
      <div className="flex flex-col md:flex-row justify-between items-center w-full px-margin-page py-stack-md max-w-[1200px] mx-auto gap-stack-sm">
        <span className="text-label-sm text-on-surface-variant uppercase tracking-wider">
          © 2024 ELEVEN MEDICAL SYSTEMS. ALL RIGHTS RESERVED.
        </span>
        <div className="flex gap-stack-sm md:gap-8">
          <a
            href="#"
            className="text-label-sm text-on-surface-variant hover:text-primary transition-colors"
          >
            Privacy
          </a>
          <a
            href="#"
            className="text-label-sm text-on-surface-variant hover:text-primary transition-colors"
          >
            Terms
          </a>
          <a
            href="#"
            className="text-label-sm text-on-surface-variant hover:text-primary transition-colors"
          >
            Support
          </a>
        </div>
      </div>
    </footer>
  );
}
