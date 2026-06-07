export interface Exercise {
  id: string;
  name: string;
  description: string;
  icon: string;
  difficulty: 'easy' | 'medium' | 'hard';
}

interface ExerciseSelectorProps {
  exercises: Exercise[];
  onSelect: (exerciseId: string) => void;
}

const DIFFICULTY_COLORS = {
  easy: 'text-primary',
  medium: 'text-on-surface',
  hard: 'text-error',
};

export function ExerciseSelector({ exercises, onSelect }: ExerciseSelectorProps) {
  return (
    <div className="space-y-4">
      <h2 className="text-headline-md text-primary text-center mb-6">
        Choose an Exercise
      </h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-2xl mx-auto">
        {exercises.map((exercise) => (
          <button
            key={exercise.id}
            onClick={() => onSelect(exercise.id)}
            className="flex items-start gap-4 p-6 bg-surface-container-lowest border border-outline-variant rounded-lg hover:border-primary hover:bg-surface-container-low transition-all text-left group"
          >
            <div className="w-12 h-12 rounded-full bg-surface-container flex items-center justify-center group-hover:bg-primary group-hover:text-on-primary transition-colors">
              <span className="material-symbols-outlined text-[24px]">{exercise.icon}</span>
            </div>
            <div className="flex-1">
              <div className="flex items-center gap-2">
                <h3 className="text-headline-md text-on-surface">{exercise.name}</h3>
                <span className={`text-label-sm uppercase ${DIFFICULTY_COLORS[exercise.difficulty]}`}>
                  {exercise.difficulty}
                </span>
              </div>
              <p className="text-body-md text-on-surface-variant mt-1">{exercise.description}</p>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
