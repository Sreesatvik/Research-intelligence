import { Link, type LinkProps } from 'react-router-dom'
import { useWorld } from '../world/worldContext'

type Props = Omit<LinkProps, 'to'> & { to: string }

// A normal link that lets the current page animate out before the route changes.
// Modified clicks (new tab, etc.) keep their default behaviour.
export default function TransitionLink({ to, onClick, ...rest }: Props) {
  const { transitionTo } = useWorld()
  return (
    <Link
      to={to}
      {...rest}
      onClick={(event) => {
        onClick?.(event)
        if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return
        event.preventDefault()
        transitionTo(to)
      }}
    />
  )
}
