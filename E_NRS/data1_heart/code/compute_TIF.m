function [T,I,F]=compute_TIF(x,T_para,I_para,F_para)
    if x<T_para(1)
        T=0;
    else
        if x<T_para(2)
            T=(x-T_para(1))/(T_para(2)-T_para(1));
        else
            if x<T_para(3)
                T=(T_para(3)-x)/(T_para(3)-T_para(2));
            else
                if x<T_para(4)
                    T=(x-T_para(3))/(T_para(4)-T_para(3));
                else
                    T=1;
                end
            end
        end
    end
    if x<I_para(1)
        I=1;
    else        
        if x<I_para(2)
            I=(I_para(2)-x)/(I_para(2)-I_para(1));
        else
            if x<I_para(3)
                I=(x-I_para(2))/(I_para(3)-I_para(2));
            else
                if x<I_para(4)
                    I=(I_para(4)-x)/(I_para(4)-I_para(3));
                else
                    I=0;
                end
            end
        end
    end
    if x<F_para(1)
        F=1;
    else
        if x< F_para(2)
            F=(F_para(2)-x)/(F_para(2)-F_para(1));
        else
            if x< F_para(3)
                F=(x-F_para(2))/(F_para(3)-F_para(2));
            else
                if x<F_para(4)
                    F=(F_para(4)-x)/(F_para(4)-F_para(3));
                else
                    F=0;
                end
            end
        end
    end
end