# Get risk of each opponent robot
# Get risk of areas (at a lower level compared to opponent robots)
#   Only apply risk to areas when a robot may be moving into that area
#   Defend area not robot when the robot is moving very quickly
#   Can use risk zones as prediction areas for likelyhood opp robots
#       move to that zone
# 
#
# How to map defenders onto offensive threats
# Always have 1 or 2 block direct shots
# Block direct shots from others
# Agressiviness tuner to choose how many robots
#   to bring to higher threat targets
#
# Always pull Max or N+1 defenders

import composite_behavior
import behavior
import constants
import robocup
import main
import enum
import math

import tactics.positions.submissive_goalie as submissive_goalie
import tactics.positions.submissive_defender as submissive_defender

import evaluation.field
import evaluation.linear_classification
import evaluation.path

class AdaptiveDefense2(composite_behavior.CompositeBehavior):

    # Weights for robot risk scores
    # [ball_dist, ball_opp_goal]
    ROBOT_RISK_WEIGHTS = [1, 1]

    # Weights for the area risk scores
    # [ball_dist, ball_goal_opp, field_pos]
    AREA_RISK_WEIGHTS = [1, 2, 3]

    # Weights / Bias for whether a opponent is a forward or winger
    # Classifier returns true if it is a winger
    WING_FORWARD_WEIGHTS = [-1, 1.8]
    WING_FORWARD_BIAS    = 0
    WING_FORWARD_CUTOFF  = 0

    class State(enum.Enum):
        # Basic blocking for right now
        # TODO: Add clearing mode
        defending = 1

    # defender_prioirities should have a length of 5 for all non-goalie robots
    def __init__(self, defender_priorities=[20,19,18,17,16]):
        super().__init__(continuous=True)

        # TODO: Change this to number of robots - 1
        if len(defender_priorities) < 2 or len(defender_priorities > 5):
            raise RuntimeError(
                "defender_priorities should have a length in between two to five inclusive")

        self.add_state(AdaptiveDefense.State.defending,
                       behavior.Behavior.State.running)
        self.add_transition(behavior.Behavior.State.start,
                            AdaptiveDefense.State.defending,
                            lambda: True, "immediately")

        goalie = submissive_goalie.SubmissiveGoalie()
        goalie.shell_id = main.root_play().goalie_id
        self.add_subbehavior(goalie, "goalie", required=False)

        for num, priority in enumerate(defender_priorities):
            defender = submissive_defender.SubmissiveDefender()
            self.add_subbehavior(defender,
                                 'defender' + str(num + 1),
                                 required=False,
                                 priority=priority)
        
        self.debug = True
        self.kick_eval = robocup.KickEvaluator(main.system_state())
        self.num_of_defenders = len(defender_priorities) 
        self.robot_classes = [] # List of tuples of is_winger (!is_forward), class score, and robot obj
        self.agressiviness = 0 # Changes how to weight positioning on the wingers

        self.kick_eval.excluded_robots.clear()

        for bot in main.our_robots():
            self.kick_eval.add_excluded_robot(bot)

    def execute_running(self):
        # Classify the opponent robots as wingers or forwards
        self.classify_opponent_robots()
        # Apply roles
        self.apply_blocking_roles()

    def get_goalie_gravity(self):
        offset = .8
        weight = 0
        for r in self.robot_classes:
            weight += r[3]
        weight /= len(self.robot_classes)
        return weight * offset

    def build_defensive_formation(self, non_direct_defenders):
        # Using the offsets and the classifier, create the formation
        # based on the number of offenders on each side of the ball
        # List the positions and what robot they are defending
        goalie_weight = self.get_goalie_gravity()
        our_goal = robocup.Point(0, 0)
        ATTACKER_CONST = 1
        DEFENDER_CONST = 1
        BALL_CONST = 1
        GOALIE_CONST = 1

        for defender in non_direct_defenders:
            defender.remove_Subehaviors()
            new_x_pos = defender.pos[0]
            new_y_pos = defender.pos[1]
            for attacker_tuple in self.robot_classes:
                new_x_pos += attacker_tuple[3] / pow(defender.pos - attacker.pos).mag(), 2) * (attacker.pos[0] - defender.pos[0]) * ATTACKER_CONST
                new_y_pos += attacker_tuple[3] / pow(defender.pos - attacker.pos).mag(), 2) * (attacker.pos[1] - defender.pos[1]) * ATTACKER_CONST
            for ally_defender in main.out_robots():
                if defender != ally_defender:
                    new_x_pos -= 1 / pow(defender.pos - ally_defender.pos).mag(), 2) * (ally_defender.pos[0] - defender.pos[0]) * DEFENDER_CONST
                    new_y_pos -= 1 / pow(defender.pos - ally_defender.pos).mag(), 2) * (ally_defender.pos[1] - defender.pos[0]) * DEFENDER_CONST
            new_x_pos += 1 / pow(defender.pos - main.ball().pos).mag(), 2) * (main.ball().pos[0] - defender.pos[0]) * BALL_CONST
            new_y_pos += 1 / pow(defender.pos - main.ball().pos).mag(), 2) * (main.ball().pos[1] - defender.pos[0]) * BALL_CONST
            
            new_x_pos -= goalie_weight / pow((defender.pos - our_goal).mag(), 2) * (0 - defender.pos[0]) * GOALIE_CONST
            new_y_pos -= goalie_weight / pow((defender.pos - our_goal).mag(), 2) * (0 - defender.pos[1]) * GOALIE_CONST

            move_point = robocup.Point(new_x_pos, new_y_pos)
            skill = skills.move.Move(move_point)

            defender.add_subbehavior(skill, "move to high danger area")

    # Applies roles to each defender
    def apply_blocking_roles(self, unassigned_handlers):
        # Assume min 2 defenders, max 5 defenders are avaliable. 
        direct_defenders = None
        current_threat = None 
        capture = False

        # This block of code assigns one robot that is responsible for direct blocking!
        # If the ball is moving, it is being passed or shot towards our goal
        if (main.ball().vel.mag() > 0.4):
            # If the ball is moving towards are goal, assume it is a shot and block it
            # TODO what if the ball is being passed to a robot in the same direction as our goal? Fix edge case
            if evaluation.ball.is_moving_towards_out_goal():
                current_threat = main.ball();
                direct_defenders = self.get_best_defender(unassigned_handlers)
            # Otherwise, the ball is being passed.
            else:
                # If we can get to the ball before the enemy, CAPTURE IT!
                if evaluation.path.can_collect_ball_before_opponent():
                    capture = True
                # Otherwise, we must directly block that bot
                current_threat = self.determine_reciever();
                direct_defenders = self.get_best_defender(unassigned_handlers)
        # Ball is NOT moving, block whichever robot has the ball
        else:
            current_threat = evaluation.ball.robot_has_ball(r) for r in main.their_robots()

        # This block of code assigns robots to dangerous areas of the field

        # This block of code assigns remaining robots to direct blocking

        # This block of code moves robots to their locations specified
        pass

    # If the ball is being passed, determine who is recieving the pass
    def determine_reciever(self): 
        ball_travel_line = robocup.Line(main.ball().pos,
                                        main.ball().pos + main.ball().vel)
        most_likely_reciever = None 
        smallest_angle = float("inf")
        for opp in main.their_robots():
            nearest_pt = ball_travel_line.nearest_point(opp.pos)
            dx = (nearest_pt - main.ball().pos).mag()
            dy = (opp.pos - nearest_pt).mag()
            angle = abs(math.atan2(dy, dx))
            if angle < smallest_angle:
                smallest_angle = angle
                most_likely_reciever = opp
        return most_likely_reciever

    # Assign the best defender to block the incoming ball that is moving towards our goal
    # Score based on how close the defender is to the goal (more reaction time)
    # And based on how much the defender is already in the way of the ball
    def get_best_defender(self, possible_defenders):
        ball_travel_line = robocup.Line(main.ball().pos,
                                        main.ball().pos + main.ball().vel)
        our_goal = robocup.Point(0, 0)
        bestDefender = None
        bestDefenderScore = float("inf") # looking to minimize the score
        bestIndex = None
        for indx, defender in enumerate(possible_defenders):
            nearest_pt = ball_travel_line.nearest_point(defender.pos)
            dx = (nearest_pt - main.ball().pos).mag()
            dy = (defender.pos - nearest_pt).mag()
            angle = abs(math.atan2(dy, dx))
            defender_score = (angle) * (defender.pos - our_goal).mag();
            if  defender_score < bestDefenderScore:
                bestDefender = defender
                bestDefenderScore = defender_score
                bestIndex = indx
        del possible_defenders[bestIndex]
        return bestDefender

    def classify_opponent_robots(self):
        # Classify opponent robots as a winger or forward
        # Wingers are more towards the outside and require more space when defending
        # Forwards mostly have the ball or are near the ball and require shot blocking

        del self.robot_classes[:]

        for bot in main.their_robots():
            if bot.visible:
                robot_risk_score = self.calculate_robot_risk_scores(bot)
                area_risk_score  = self.calculate_area_risk_scores(bot)
                shooting_risk_score = self.chance_to_score(bot)

                features = [robot_risk_score, area_risk_score]

                is_wing, class_score = evaluation.linear_classification.binary_classification(features,
                                            AdaptiveDefense.WING_FORWARD_WEIGHTS,
                                            AdaptiveDefense.WING_FORWARD_BIAS,
                                            AdaptiveDefense.WING_FORWARD_CUTOFF)
                
                self.robot_classes.append((is_wing, class_score, bot, robot_risk_score + area_risk_score + shooting_risk_score))

                if self.debug and is_wing:
                    main.system_state().draw_circle(bot.pos, 0.5, constants.Colors.White, "Defense: Class Wing")
                elif self.debug and not is_wing:
                    main.system_state().draw_circle(bot.pos, 0.5, constants.Colors.Black, "Defense: Class Forward")


    def calculate_robot_risk_scores(self, bot):
        max_dist = robocup.Point(constants.Field.Length, constants.Field.Width).mag()
        our_goal = robocup.Point(0, 0)
        dist_sens = 1.5
        ball_opp_sens = 1.5

        # How far away the robot is from the ball, closer is higher
        ball_dist = pow(1 - dist_sens*(bot.pos- main.ball().pos).mag() / max_dist, 2)
        # How large the angle is between the ball, opponent, and goal, smaller angle is better
        ball_opp_goal = math.pow((math.fabs((main.ball().pos - bot.pos).angle_between(bot.pos - our_goal)) / math.pi), ball_opp_sens)

        risk_score = AdaptiveDefense.ROBOT_RISK_WEIGHTS[0] * ball_dist + \
                     AdaptiveDefense.ROBOT_RISK_WEIGHTS[1] * ball_opp_goal

        risk_score /= sum(AdaptiveDefense.ROBOT_RISK_WEIGHTS)

        if self.debug:
            main.system_state().draw_text("Robot Risk: " + str(int(risk_score*100)), 
                bot.pos - robocup.Point(0, 0.25), constants.Colors.White, "Defense: Risk")
        
        return risk_score
    
    # Danger based purely on the location of an enemy
    def calculate_area_risk_scores(self, bot):
        max_dist = robocup.Point(constants.Field.Length, constants.Field.Width).mag()
        our_goal = robocup.Point(0, 0)
        ball_goal_sens = 2.5
        dist_sens = 1.5

        # How far away the robot is from the ball, further is higher
        ball_dist = 1 - pow(1 - dist_sens*(bot.pos - main.ball().pos).mag() / max_dist, 2)
        # How large the angle is between the ball, goal, and opponent, smaller angle is better
        ball_goal_opp = 1 - math.pow(math.fabs((main.ball().pos - our_goal).angle_between(our_goal - bot.pos)) / math.pi, ball_goal_sens)
        # Location on the field based on closeness to the goal line, closer is better
        field_pos = evaluation.field.field_pos_coeff_at_pos(bot.pos, 0, 1, 0, False)

        risk_score = AdaptiveDefense.AREA_RISK_WEIGHTS[0] * ball_dist + \
                     AdaptiveDefense.AREA_RISK_WEIGHTS[1] * ball_goal_opp + \
                     AdaptiveDefense.AREA_RISK_WEIGHTS[2] * field_pos

        risk_score /= sum(AdaptiveDefense.AREA_RISK_WEIGHTS)

        if self.debug:
            main.system_state().draw_text("Area Risk: " + str(int(risk_score*100)), 
                bot.pos + robocup.Point(0, 0.25), constants.Colors.White, "Defense: Risk")

        return risk_score

    def chance_to_score(self, bot, excluded_bots = []):
        excluded_bots.append(bot)

        passChance = evaluation.passing.eval_pass(
            main.ball().pos, bot.pos, excluded_robots=excluded_bots)

        self.kick_eval.excluded_robots.clear()
        for r in excluded_bots:
            self.kick_eval.add_excluded_robot(r)

        point, shotChance = self.kick_eval.eval_pt_to_our_goal(bot.pos)

        return passChance * shotChance