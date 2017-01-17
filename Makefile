NAME		= pathtracer
NVCC 		= nvcc
NVCCFLAGS	= -std=c++11 -dc -I /usr/local/cuda/include -I include -O3
CXXFLAGS	= -std=c++11 -Wall -Wextra -I /usr/local/cuda/include -I include -O3

LDFLAGS		= -Xlinker -framework,SDL2

CXXFILES	=	Window.cpp			\
				Camera.cpp

CUFILES		=	main.cu				\
				Ray.cu				\
				Scene.cu			\
				Object.cu			\
				Sphere.cu			\
				Plane.cu			\
				Light.cu

CXXSRC		= $(addprefix $(CXXFILES), source/)
CUSRC		= $(addprefix $(CUFILES), source/)
CXXOBJ		= $(CXXFILES:%.cpp=obj/%.o)
CUOBJ		= $(CUFILES:%.cu=obj/%.o)

NO_COLOR	= \x1b[0m
GREEN		= \x1b[32;01m
RED			= \x1b[31;01m
YELLOW		= \x1b[33;01m
GRAY		= \x1b[37;01m

.PHONY: all re clean fclean

all: $(NAME)

$(NAME): $(CXXOBJ) $(CUOBJ)
	@printf "$(GRAY)Building $(RED)$(NAME)$(NO_COLOR)"
	@$(NVCC) $(CXXOBJ) $(CUOBJ) -o $(NAME) $(LDFLAGS)
	@if [ -f $(NAME) ] ; \
	then \
		printf " $(GREEN)✔$(NO_COLOR)\n" ; \
	fi;

obj/%.o: source/%.cpp
	@mkdir -p obj
	@printf "$(YELLOW)$<$(NO_COLOR) "
	@$(CXX) $(CXXFLAGS) -c $< -o $@
	@if [ -f $@ ] ; \
	then \
		printf "$(GREEN)✔$(NO_COLOR)\n" ; \
	fi;

obj/%.o: source/%.cu
	@mkdir -p obj
	@printf "$(YELLOW)$<$(NO_COLOR) "
	@$(NVCC) $(NVCCFLAGS) -c $< -o $@
	@if [ -f $@ ] ; \
	then \
		printf "$(GREEN)✔$(NO_COLOR)\n" ; \
	fi;

clean:
	@printf "$(GRAY)Removing objects$(NO_COLOR)"
	@rm -rf obj
	@printf " $(GREEN)✔$(NO_COLOR)\n"

fclean: clean
	@printf "$(GRAY)Removing $(RED)$(NAME)$(NO_COLOR)"
	@rm -f $(NAME)
	@printf " $(GREEN)✔$(NO_COLOR)\n"

re: fclean all
