parent(/oedipus, /antigone).
parent(/oedipus, /ismene).
parent(/oedipus, /eteocles).
parent(/oedipus, /polynices).

sibling(Person1, Person2) :-
  parent(P, Person1), parent(P, Person2), Person1 != Person2.

::print sibling.
