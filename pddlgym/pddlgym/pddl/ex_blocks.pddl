;; Authors: Michael Littman and David Weissman
;; Modified by: Blai Bonet
;; Comment: Good plans are those that avoid putting blocks on table since the probability of detonation is higher

(define (domain exploding-blocksworld)
  (:requirements :typing :conditional-effects :probabilistic-effects)
  (:types block)
  (:predicates (on ?b1 - block ?b2 - block) (on-table ?b - block) (clear ?b - block) (holding ?b - block) (emptyhand) (no-detonated ?b - block) (no-destroyed ?b - block) (no-destroyed-table)
  (pickup ?b - block)
  (pickupfromtable ?b - block)
  (putdown)
  (putonblock ?b - block))

  ; (:actions pickup pickupfromtable putdown putonblock)

  (:action pick-up
   :parameters (?b1 - block ?b2 - block)
   :precondition (and (pickup ?b1) (emptyhand) (clear ?b1) (on ?b1 ?b2) (no-destroyed ?b1))
   :effect (and (holding ?b1) (clear ?b2) (not (emptyhand)) (not (on ?b1 ?b2)))
  )
  (:action pick-up-from-table
   :parameters (?b - block)
   :precondition (and (pickupfromtable ?b) (emptyhand) (clear ?b) (on-table ?b) (no-destroyed ?b))
   :effect (and (holding ?b) (not (emptyhand)) (not (on-table ?b)))
  )
  (:action put-down
   :parameters (?b - block)
   :precondition (and (putdown) (holding ?b) (no-destroyed-table))
   :effect (and (emptyhand) (on-table ?b) (not (holding ?b))
                (probabilistic 2/5 (and (when (no-detonated ?b) (and (not (no-destroyed-table)) (not (no-detonated ?b)))))))
  )
  (:action put-on-block
   :parameters (?b1 - block ?b2 - block)
   :precondition (and (putonblock ?b2) (not (= ?b1 ?b2)) (holding ?b1) (clear ?b2) (no-destroyed ?b2))
   :effect (and (emptyhand) (on ?b1 ?b2) (not (holding ?b1)) (not (clear ?b2))
                (probabilistic 1/10 (and (when (no-detonated ?b1) (and (not (no-destroyed ?b2)) (not (no-detonated ?b1)))))))
  )
)

