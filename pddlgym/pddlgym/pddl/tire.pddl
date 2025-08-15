;;;  Authors: Michael Littman and David Weissman  ;;;
;;;  Modified: Blai Bonet for IPC 2006 ;;;

(define (domain tire)
  (:requirements :typing :strips :probabilistic-effects)
  (:types location)
  (:predicates 
    (vehicle-at ?loc - location) 
    (spare-in ?loc - location) 
    (road ?from - location ?to - location) 
    (not-flattire) 
    (hasspare)
    (movecar ?to - location)
    (loadtire)
    (changetire)
  )
  ; (:actions movecar loadtire changetire)
  (:action move-car
    :parameters (?from - location ?to - location)
    :precondition (and (movecar ?to) (vehicle-at ?from) (road ?from ?to) (not-flattire))
    :effect (and (vehicle-at ?to) (not (vehicle-at ?from)) (probabilistic 2/5 (not (not-flattire))))
  )
  (:action loadtire
    :parameters (?loc - location)
    :precondition (and (loadtire) (vehicle-at ?loc) (spare-in ?loc))
    :effect (and (hasspare) (not (spare-in ?loc)))
  )
  (:action changetire
    :parameters 
    :precondition (and (hasspare) (changetire))
    :effect (and (probabilistic 1/2 (and (not (hasspare)) (not-flattire))))
  )
)
