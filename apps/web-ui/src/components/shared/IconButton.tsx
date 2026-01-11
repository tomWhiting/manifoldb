import { type ComponentProps, type ReactNode, forwardRef } from 'react'

interface IconButtonProps extends ComponentProps<'button'> {
  icon: ReactNode
  tooltip?: string
  size?: 'sm' | 'md' | 'lg'
  variant?: 'ghost' | 'default'
}

export const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(
  ({ icon, tooltip, size = 'md', variant = 'ghost', className = '', ...props }, ref) => {
    const sizeClasses = {
      sm: 'w-6 h-6',
      md: 'w-8 h-8',
      lg: 'w-10 h-10',
    }

    const variantClasses = {
      ghost: 'hover:bg-bg-tertiary text-text-muted hover:text-text-secondary',
      default: 'bg-bg-tertiary hover:bg-border text-text-secondary',
    }

    return (
      <button
        ref={ref}
        title={tooltip}
        className={`
          inline-flex items-center justify-center rounded-md
          transition-colors duration-150
          focus:outline-none focus:ring-2 focus:ring-accent/50
          disabled:opacity-50 disabled:cursor-not-allowed
          ${sizeClasses[size]}
          ${variantClasses[variant]}
          ${className}
        `}
        {...props}
      >
        {icon}
      </button>
    )
  }
)

IconButton.displayName = 'IconButton'
