(define (domain prob_blocks)
  (:requirements :probabilistic-effects :typing)
  (:types block)
  (:predicates (holding ?b - block) (emptyhand) (on-table ?b - block) (on ?b1 - block ?b2 - block) (clear ?b - block)
    (pickup ?b - block) 
    (pickupfromtable ?b - block)
    (putonblock ?b - block)
    (picktower ?b - block)
    (puttoweronblock ?b - block)
    (putdown))
  ; (:actions pickup pickupfromtable putonblock putdown picktower puttoweronblock)
  (:action pick-up
    :parameters (?b1 - block ?b2 - block)
    :precondition (and (pickup ?b1) (emptyhand) (clear ?b1) (on ?b1 ?b2))
    :effect
      (and (probabilistic
        0.75 (and (holding ?b1) (clear ?b2) (not (emptyhand)) (not (on ?b1 ?b2)))
        0.25 (and (clear ?b2) (on-table ?b1) (not (on ?b1 ?b2)))))
  )
  (:action pick-up-from-table
    :parameters (?b - block)
    :precondition (and (pickupfromtable ?b) (emptyhand) (clear ?b) (on-table ?b))
    :effect (and (probabilistic 0.75 (and (holding ?b) (not (emptyhand)) (not (on-table ?b)))))
  )
  (:action put-on-block
    :parameters (?b1 - block ?b2 - block)
    :precondition (and (putonblock ?b2) (holding ?b1) (not (= ?b1 ?b2)) (clear ?b1) (clear ?b2))
    :effect (and (probabilistic 0.75 (and (on ?b1 ?b2) (emptyhand) (not (holding ?b1)) (not (clear ?b2)))
                           0.25 (and (on-table ?b1) (emptyhand) (not (holding ?b1)))))
  )
  (:action put-down
    :parameters (?b - block)
    :precondition (and (putdown) (holding ?b))
    :effect (and (on-table ?b) (emptyhand) (not (holding ?b)))
  )
  (:action pick-tower
    :parameters (?b1 - block ?b2 - block)
    :precondition (and (picktower ?b1) (emptyhand) (not (clear ?b1)) (on ?b1 ?b2))
    :effect
      (and (probabilistic 0.1 (and (holding ?b1) (clear ?b2) (not (emptyhand)) (not (on ?b1 ?b2)))))
  )
  (:action put-tower-on-block
    :parameters (?b1 - block ?b2 - block)
    :precondition (and (puttoweronblock ?b2) (holding ?b1) (not (clear ?b1)) (clear ?b2))
    :effect (and (probabilistic 0.1 (and (on ?b1 ?b2) (emptyhand) (not (holding ?b1)) (not (clear ?b2)))
                           0.9 (and (on-table ?b1) (emptyhand) (not (holding ?b1)))))
  )
)
